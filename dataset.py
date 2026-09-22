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
    ├── images/           # H-alpha FITS files
    ├── masks/            # Ground truth binary masks
    └── annotations.json  # Optional metadata
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
        
        # Check if using cached .npy format
        cache_dir = self.data_root / "images"
        if self.use_cache and cache_dir.exists():
            npy_files = list(cache_dir.glob("*.npy"))
            if npy_files:
                self.use_npy_cache = True
                self.image_files = sorted(npy_files)
                self.mask_files = []
                if (self.data_root / "masks").exists():
                    self.mask_files = sorted((self.data_root / "masks").glob("*.npy"))
                print(f"Using cached .npy format: {len(self.image_files)} images")
            else:
                self.use_npy_cache = False
        else:
            self.use_npy_cache = False
            
        if not self.use_npy_cache:
            # Look for FITS files (H-alpha images)
            self.image_dir = self.data_root / "images"
            self.mask_dir = self.data_root / "masks"
            
            if not self.image_dir.exists():
                raise ValueError(f"Image directory not found: {self.image_dir}")
            
            # Get all image files
            image_extensions = ['.fits', '.fit', '.png', '.jpg', '.jpeg']
            self.image_files = []
            for ext in image_extensions:
                self.image_files.extend(self.image_dir.glob(f"*{ext}"))
            
            self.image_files = sorted(self.image_files)
            
            if split == 'train' and self.mask_dir.exists():
                mask_extensions = ['.png', '.jpg', '.jpeg', '.npy']
                self.mask_files = []
                for ext in mask_extensions:
                    self.mask_files.extend(self.mask_dir.glob(f"*{ext}"))
                self.mask_files = sorted(self.mask_files)
            else:
                self.mask_files = []
        
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
    
    def load_mask(self, mask_path: Optional[Path]) -> Optional[np.ndarray]:
        """Load a mask from file (supports PNG, NPY)."""
        if mask_path is None or not mask_path.exists():
            return None
            
        if mask_path.suffix.lower() == '.npy':
            mask = np.load(mask_path).astype(np.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.float32) / 255.0
            
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
        if self.split == 'train' and self.mask_files and idx < len(self.mask_files):
            mask_data = self.load_mask(self.mask_files[idx])
        
        # Preprocess
        processed = preprocess_observation(image_data)
        
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
                    'mask': torch.from_numpy(gt_tile).float(),
                    'has_filament': has_fil,
                    'tile_coords': (y, x),
                    'image_id': image_path.stem
                }
            else:
                # Fallback if no valid tiles
                sample = {
                    'image': torch.from_numpy(processed.image).float(),
                    'valid_mask': torch.from_numpy(processed.valid_mask).float(),
                    'mask': torch.zeros_like(processed.valid_mask),
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
                      use_cache: bool = True) -> Tuple[torch.utils.data.DataLoader, 
                                                      Optional[torch.utils.data.DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Path to the dataset root directory
        batch_size: Batch size for training
        tile_size: Size of tiles for training
        num_workers: Number of worker processes for data loading
        use_cache: Whether to use cached .npy files
        
    Returns:
        train_loader, val_loader (val_loader is None if no validation split)
    """
    train_dataset = SolarFilamentDataset(
        data_root=data_root,
        split='train',
        tile_size=tile_size,
        use_cache=use_cache
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # For now, use the same dataset for validation
    # In practice, you'd want a separate validation split
    val_loader = None
    
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