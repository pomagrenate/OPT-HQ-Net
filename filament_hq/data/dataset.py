"""
PyTorch Dataset for Filament-HQ Tiled Training.

Loads raw 2048x2048 H-alpha images and pre-rendered masks, applies 4-channel
physical preprocessing, and samples 1024x1024 foreground-centric tile patches
with zero spatial downsampling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from filament_hq.data.preprocessor import SolarPhysicalPreprocessor
from filament_hq.data.tiler import ImageTiler

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class FilamentTileDataset(Dataset):
    """
    Filament-HQ Dataset for 1024x1024 tile training with 4-channel preprocessing.

    Parameters
    ----------
    data_root : str | Path
        Path to dataset folder containing images/ and masks/
    tile_size : int
        Fixed input tile contract (default 1024).
    fg_prob : float
        Probability of centering tile crop on a random foreground filament.
    augment : bool
        Apply geometric and photometric augmentations.
    overfit_single_image : bool
        If True, locks dataset to 1 single image for Phase 0 verification.
    """

    def __init__(
        self,
        data_root: str | Path,
        tile_size: int = 1024,
        fg_prob: float = 0.8,
        augment: bool = True,
        overfit_single_image: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.tile_size = tile_size
        self.fg_prob = fg_prob
        self.augment = augment
        self.overfit_single_image = overfit_single_image
        self.preprocessor = SolarPhysicalPreprocessor()

        # Locate image directory
        if (self.data_root / "images").exists():
            self.image_dir = self.data_root / "images"
        elif (self.data_root / "train_images").exists():
            self.image_dir = self.data_root / "train_images"
        else:
            self.image_dir = self.data_root

        self.mask_dir = self.data_root / "masks"
        exts = [".png", ".jpg", ".jpeg", ".fits"]

        self.image_files = sorted(
            [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        )

        if not self.image_files:
            raise FileNotFoundError(f"No image files found in '{self.image_dir}'.")

        if self.overfit_single_image:
            self.image_files = [self.image_files[0]]
            print(f"[Dataset] Locked to single overfit image: {self.image_files[0].name}")

    def __len__(self) -> int:
        return len(self.image_files) * (500 if self.overfit_single_image else 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = 0 if self.overfit_single_image else (idx % len(self.image_files))
        img_path = self.image_files[real_idx]
        img_id = img_path.stem

        # 1. Load image (2048x2048)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # 2. Load masks (2048x2048)
        mask_npz = self.mask_dir / f"{img_id}.npz"
        if mask_npz.exists():
            data = np.load(mask_npz)
            masks_arr = data["masks"]  # (N, H, W)
        else:
            masks_arr = np.zeros((0, img.shape[0], img.shape[1]), dtype=np.uint8)

        # 3. Apply 4-channel preprocessor
        ch4_img = self.preprocessor(img)  # (H, W, 4)
        h, w = ch4_img.shape[:2]

        # Combine instance masks into unified semantic mask and instance map
        semantic_mask = (masks_arr.sum(axis=0) > 0).astype(np.uint8) if len(masks_arr) > 0 else np.zeros((h, w), dtype=np.uint8)
        
        # 4. Foreground-Centric Tile Crop (1024x1024)
        crop_y1, crop_x1 = self._sample_crop_coords(semantic_mask, h, w)
        crop_y2 = crop_y1 + self.tile_size
        crop_x2 = crop_x1 + self.tile_size

        tile_img = ch4_img[crop_y1:crop_y2, crop_x1:crop_x2]  # (1024, 1024, 4)
        tile_semantic = semantic_mask[crop_y1:crop_y2, crop_x1:crop_x2]  # (1024, 1024)
        tile_masks = masks_arr[:, crop_y1:crop_y2, crop_x1:crop_x2] if len(masks_arr) > 0 else np.zeros((0, self.tile_size, self.tile_size), dtype=np.uint8)

        # Filter masks present in crop
        valid_idx = [i for i in range(len(tile_masks)) if tile_masks[i].sum() > 5]
        tile_masks = tile_masks[valid_idx] if valid_idx else np.zeros((0, self.tile_size, self.tile_size), dtype=np.uint8)

        # 5. Derive Boundary and Skeleton Maps for Tile
        tile_boundary = self._compute_boundary_map(tile_semantic)
        tile_skeleton = self._compute_skeleton_map(tile_semantic)

        # 6. Apply Augmentation if enabled
        if self.augment and not self.overfit_single_image:
            tile_img, tile_semantic, tile_boundary, tile_skeleton, tile_masks = self._apply_augmentations(
                tile_img, tile_semantic, tile_boundary, tile_skeleton, tile_masks
            )

        # 7. Convert to PyTorch Tensors
        # Convert HWC float32 [1024, 1024, 4] -> CHW [4, 1024, 1024]
        img_tensor = torch.from_numpy(tile_img).permute(2, 0, 1).float()
        sem_tensor = torch.from_numpy(tile_semantic).unsqueeze(0).float()
        bnd_tensor = torch.from_numpy(tile_boundary).unsqueeze(0).float()
        skl_tensor = torch.from_numpy(tile_skeleton).unsqueeze(0).float()
        inst_tensor = torch.from_numpy(tile_masks).float()  # (N, 1024, 1024)

        return {
            "image": img_tensor,            # (4, 1024, 1024)
            "semantic": sem_tensor,        # (1, 1024, 1024)
            "boundary": bnd_tensor,        # (1, 1024, 1024)
            "skeleton": skl_tensor,        # (1, 1024, 1024)
            "instances": inst_tensor,      # (N, 1024, 1024)
            "image_id": img_id,
        }

    def _sample_crop_coords(self, sem_mask: np.ndarray, h: int, w: int) -> Tuple[int, int]:
        """Sample top-left (y1, x1) for 1024x1024 tile crop."""
        if h <= self.tile_size or w <= self.tile_size:
            return 0, 0

        max_y = h - self.tile_size
        max_x = w - self.tile_size

        if np.random.rand() < self.fg_prob and sem_mask.sum() > 0:
            fg_ys, fg_xs = np.where(sem_mask > 0)
            pick_idx = np.random.choice(len(fg_ys))
            cy, cx = fg_ys[pick_idx], fg_xs[pick_idx]
            
            y1 = int(np.clip(cy - self.tile_size // 2 + np.random.randint(-100, 100), 0, max_y))
            x1 = int(np.clip(cx - self.tile_size // 2 + np.random.randint(-100, 100), 0, max_x))
            return y1, x1

        y1 = np.random.randint(0, max_y + 1)
        x1 = np.random.randint(0, max_x + 1)
        return y1, x1

    @staticmethod
    def _compute_boundary_map(sem_mask: np.ndarray) -> np.ndarray:
        """Derive 1-pixel boundary map via Laplace filter."""
        if sem_mask.sum() == 0:
            return np.zeros_like(sem_mask, dtype=np.float32)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        bnd = cv2.filter2D(sem_mask.astype(np.float32), -1, kernel)
        return (bnd > 0.1).astype(np.float32)

    @staticmethod
    def _compute_skeleton_map(sem_mask: np.ndarray) -> np.ndarray:
        """Derive topological 1D skeleton map."""
        if sem_mask.sum() == 0 or not SKIMAGE_AVAILABLE:
            return np.zeros_like(sem_mask, dtype=np.float32)
        skel = skeletonize(sem_mask.astype(bool))
        return skel.astype(np.float32)

    def _apply_augmentations(self, img, sem, bnd, skl, masks):
        """Apply aligned geometric flips and 90-deg rotations."""
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            img = np.rot90(img, k, (0, 1))
            sem = np.rot90(sem, k, (0, 1))
            bnd = np.rot90(bnd, k, (0, 1))
            skl = np.rot90(skl, k, (0, 1))
            if len(masks) > 0:
                masks = np.rot90(masks, k, (1, 2))

        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            sem = np.fliplr(sem)
            bnd = np.fliplr(bnd)
            skl = np.fliplr(skl)
            if len(masks) > 0:
                masks = np.fliplr(masks)

        if np.random.rand() > 0.5:
            img = np.flipud(img)
            sem = np.flipud(sem)
            bnd = np.flipud(bnd)
            skl = np.flipud(skl)
            if len(masks) > 0:
                masks = np.flipud(masks)

        return img.copy(), sem.copy(), bnd.copy(), skl.copy(), masks.copy()
