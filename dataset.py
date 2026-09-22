"""
Dataset module for loading MAGFiLO data.

Supports loading H-alpha solar images and their corresponding filament masks
from the MAGFiLO_1.0_Kaggle_2026 dataset structure.
"""

from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from astropy.io import fits
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False

from preprocessing import preprocess_observation, extract_tiles, class_balanced_sample


class SolarFilamentDataset(Dataset):
    """
    Dataset for loading MAGFiLO solar filament data.
    
    Expected directory structure:
    data_root/
    ├── train_images/      # H-alpha JPEG files (training)
    ├── test_images/       # H-alpha JPEG files (testing)
    └── MAGFiLO_1.0_Annotations_kaggle2026_train.json  # Training annotations
    """
    
    def __init__(self, data_root: str, split: str = 'train', 
                 tile_size: int = 256, overlap: float = 0.25,
                 use_cache: bool = True, transform=None):
        """
        Args:
            data_root: Path to the dataset root directory
            split: 'train' or 'test'
            tile_size: Size of tiles for training
            overlap: Overlap fraction for tiling
            use_cache: Whether to use cached .npy files if available
            transform: Optional transform function
        """
        self.data_root = Path(data_root)
        self.split = split
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_cache = use_cache
        self.transform = transform
        
        # Determine image directory based on split
        # Handle both directory structures:
        # 1. data_root/train/train_images (MAGFiLO structure)
        # 2. data_root/train_images (flat structure)
        if split == 'train':
            # Try MAGFiLO structure first
            possible_dirs = [
                self.data_root / "train" / "train_images",
                self.data_root / "train_images",
                self.data_root / "train"
            ]
            
            for possible_dir in possible_dirs:
                if possible_dir.exists():
                    self.image_dir = possible_dir
                    break
            else:
                raise ValueError(f"Image directory not found in any of: {possible_dirs}")
            
            # Try to find annotations file
            possible_annotations = [
                self.data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                self.data_root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                self.data_root / "annotations.json"
            ]
            
            for possible_file in possible_annotations:
                if possible_file.exists():
                    self.annotations_file = possible_file
                    break
            else:
                self.annotations_file = None
                print("Warning: No annotations file found")
        else:
            # Test split
            possible_dirs = [
                self.data_root / "test" / "test_images",
                self.data_root / "test_images",
                self.data_root / "test"
            ]
            
            for possible_dir in possible_dirs:
                if possible_dir.exists():
                    self.image_dir = possible_dir
                    break
            else:
                raise ValueError(f"Image directory not found in any of: {possible_dirs}")
            
            self.annotations_file = None
        
        # Check if using cached .npy format
        if self.use_cache and self.image_dir.exists():
            npy_files = list(self.image_dir.glob("*.npy"))
            if npy_files:
                self.use_npy_cache = True
                self.image_files = sorted(npy_files)
                print(f"Using cached .npy format: {len(self.image_files)} images")
            else:
                self.use_npy_cache = False
        else:
            self.use_npy_cache = False
            
        if not self.use_npy_cache:
            # Look for image files
            if not self.image_dir.exists():
                raise ValueError(f"Image directory not found: {self.image_dir}")
            
            # Get all image files (support JPEG, PNG, FITS)
            image_extensions = ['.jpeg', '.jpg', '.png', '.fits', '.fit']
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(self.image_dir.glob(f"*{ext}"))
            
            self.image_files = sorted(self.image_files)
            
            # Load annotations for training
            self.annotations = None
            if split == 'train' and self.annotations_file.exists():
                import json
                with open(self.annotations_file, 'r') as f:
                    self.annotations = json.load(f)
                print(f"Loaded annotations with {len(self.annotations.get('annotations', []))} entries")
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image from file (supports FITS, PNG, NPY)."""
        if image_path.suffix.lower() in ['.fits', '.fit']:
            if not _HAS_ASTROPY:
                raise ImportError("astropy is required to load FITS files")
            with fits.open(image_path) as hdul:
                data = hdul[0].data.astype(np.float32)
        elif image_path.suffix.lower() == '.npy':
            data = np.load(image_path).astype(np.float32)
        else:
            # Assume standard image format
            img = Image.open(image_path).convert('L')
            data = np.array(img, dtype=np.float32)
            
        return data
    
    def load_mask_from_annotations(self, image_name: str) -> Optional[np.ndarray]:
        """Load mask from COCO-format JSON annotations based on image name (optimized)."""
        if self.annotations is None:
            return None
        
        # Check cache first
        cache_key = f"{image_name}_mask"
        if hasattr(self, '_mask_cache') and cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        # Initialize cache if not exists
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        
        # Find the image ID for this image
        image_id = None
        for img_info in self.annotations.get('images', []):
            if img_info.get('file_name') == image_name:
                image_id = img_info.get('id')
                break
        
        if image_id is None:
            return None
        
        # Find all annotations for this image
        annotations = []
        for annotation in self.annotations.get('annotations', []):
            if annotation.get('image_id') == image_id:
                annotations.append(annotation)
        
        if not annotations:
            self._mask_cache[cache_key] = None
            return None
        
        # Create a binary mask from polygon annotations
        # Get image dimensions
        img_info = next((img for img in self.annotations.get('images', []) if img.get('id') == image_id), None)
        if img_info is None:
            return None
        
        height = img_info.get('height', 2048)
        width = img_info.get('width', 2048)
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Fill mask with polygons - optimized using OpenCV
        try:
            import cv2
            for annotation in annotations:
                if 'segmentation' in annotation:
                    polygons = annotation['segmentation']
                    for polygon in polygons:
                        # Reshape polygon to (N, 1, 2) for OpenCV
                        poly_points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                        # Fill polygon directly on numpy array (much faster than PIL)
                        cv2.fillPoly(mask, [poly_points], 1)
        except Exception as e:
            print(f"Warning: Could not load mask for {image_name}: {e}")
            self._mask_cache[cache_key] = None
            return None
        
        # Cache the result
        self._mask_cache[cache_key] = mask
        
        return mask
    
    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_files[idx]
        
        # Load image
        if self.use_npy_cache:
            image_data = np.load(image_path).astype(np.float32)
        else:
            image_data = self.load_image(image_path)
        
        # Load mask if available (training mode)
        mask_data = None
        if self.split == 'train':
            mask_data = self.load_mask_from_annotations(image_path.name)
        
        # Simplified preprocessing for speed (skip full preprocessing for now)
        # Just normalize and stack with a simple ridge prior
        img = image_data.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        
        # Simple normalization
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        
        # Simple ridge prior (gradient magnitude)
        import cv2
        gray = (img * 255).astype(np.uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ridge = np.sqrt(grad_x**2 + grad_y**2)
        ridge = ridge / (ridge.max() + 1e-6)
        
        # Stack
        stack = np.stack([img, ridge], axis=0).astype(np.float32)
        
        # Simple valid mask (all pixels valid for now)
        valid_mask = np.ones((1, img.shape[0], img.shape[1]), dtype=np.float32)
        
        processed = type('obj', (object,), {
            'image': stack,
            'valid_mask': valid_mask,
            'disk': (img.shape[1]//2, img.shape[0]//2, min(img.shape)//2)
        })()
        
        # Extract tiles for training
        if mask_data is not None:
            tiles = extract_tiles(
                processed.image, 
                processed.valid_mask, 
                mask_data,
                tile=self.tile_size,
                overlap=self.overlap
            )
            
            # Class-balanced sampling
            sampled_tiles = class_balanced_sample(tiles, n=1)
            if sampled_tiles:
                img_tile, valid_tile, gt_tile, has_fil, (y, x) = sampled_tiles[0]
                
                sample = {
                    'image': torch.from_numpy(img_tile).float(),
                    'valid_mask': torch.from_numpy(valid_tile).float(),
                    'mask': torch.from_numpy(gt_tile).float().unsqueeze(0),  # Add channel dimension
                    'has_filament': has_fil,
                    'tile_coords': (y, x),
                    'image_id': image_path.stem
                }
            else:
                # Fallback if no valid tiles - create a zero mask with proper shape
                sample = {
                    'image': torch.from_numpy(processed.image).float(),
                    'valid_mask': torch.from_numpy(processed.valid_mask).float(),
                    'mask': torch.zeros(1, *processed.valid_mask.shape).float(),  # Add channel dimension
                    'has_filament': False,
                    'tile_coords': (0, 0),
                    'image_id': image_path.stem
                }
        else:
            # Inference mode - return full image
            sample = {
                'image': torch.from_numpy(processed.image).float(),
                'valid_mask': torch.from_numpy(processed.valid_mask).float(),
                'mask': None,
                'image_id': image_path.stem,
                'disk': processed.disk
            }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_dataloaders(data_root: str, batch_size: int = 4, 
                      tile_size: int = 256, num_workers: int = 2,
                      use_cache: bool = True, val_split: float = 0.1) -> Tuple[torch.utils.data.DataLoader, 
                                                                                Optional[torch.utils.data.DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Path to the dataset root directory
        batch_size: Batch size for training
        tile_size: Size of tiles for training
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use cached .npy files
        val_split: Fraction of data to use for validation
        
    Returns:
        train_loader, val_loader
    """
    full_dataset = SolarFilamentDataset(
        data_root=data_root,
        split='train',
        tile_size=tile_size,
        use_cache=use_cache
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    return train_loader, val_loader


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate function for batching variable-sized samples."""
    if len(batch) == 0:
        return {}
    
    # Check if this is training mode (has masks) or inference mode
    has_masks = batch[0].get('mask') is not None
    
    if has_masks:
        images = torch.stack([item['image'] for item in batch])
        valid_masks = torch.stack([item['valid_mask'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        return {
            'image': images,
            'valid_mask': valid_masks,
            'mask': masks,
            'image_ids': [item['image_id'] for item in batch]
        }
    else:
        # Inference mode - batch might have different sizes
        return {
            'image': [item['image'] for item in batch],
            'valid_mask': [item['valid_mask'] for item in batch],
            'image_ids': [item['image_id'] for item in batch],
            'disk': [item.get('disk') for item in batch]
        }


if __name__ == "__main__":
    # Test the dataset
    print("Testing SolarFilamentDataset...")
    
    # You can test with your actual data path
    # data_root = "path/to/MAGFiLO_1.0_Kaggle_2026/train"
    # dataset = SolarFilamentDataset(data_root, split='train')
    # print(f"Dataset size: {len(dataset)}")
    # sample = dataset[0]
    # print(f"Sample keys: {sample.keys()}")
    # print(f"Image shape: {sample['image'].shape}")
    # if sample['mask'] is not None:
    #     print(f"Mask shape: {sample['mask'].shape}")
    
    print("Dataset module loaded successfully!")