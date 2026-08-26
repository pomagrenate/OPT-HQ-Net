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
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.data_root if (self.data_root / "index.pkl").exists() else None)
        self.tile_size = tile_size
        self.fg_prob = fg_prob
        self.augment = augment
        self.overfit_single_image = overfit_single_image
        self.preprocessor = SolarPhysicalPreprocessor()

        # Check if loading from precomputed offline cache
        self.index_meta = None
        if self.cache_dir and (self.cache_dir / "index.pkl").exists():
            import pickle
            index_path = self.cache_dir / "index.pkl"
            print(f"[Dataset] Loading fast precomputed cache index from: {index_path}")
            with open(index_path, "rb") as f:
                self.index_meta = pickle.load(f)["items"]
            self.image_ids = list(self.index_meta.keys())
            self.image_files = [Path(v["npy_path"]) for v in self.index_meta.values()]
        else:
            # Flexible image directory resolution
            if (self.data_root / "images").exists():
                self.image_dir = self.data_root / "images"
            elif (self.data_root / "train_images").exists():
                self.image_dir = self.data_root / "train_images"
            elif (self.data_root / "test_images").exists():
                self.image_dir = self.data_root / "test_images"
            else:
                self.image_dir = self.data_root

            self.mask_dir = self.data_root / "masks"
            exts = [".png", ".jpg", ".jpeg", ".fits"]

            self.image_files = sorted(
                [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            )

            if not self.image_files:
                raise FileNotFoundError(f"No image files found in '{self.image_dir}'.")

            # ── COCO JSON Parsing Support ─────────────────────────────────────
            self.coco_anns: Dict[str, List[Dict]] = {}
            json_files = [j for j in (list(self.data_root.glob("*.json")) + list(self.data_root.rglob("*.json"))) if j.name.lower() != "manifest.json"]
            if json_files:
                coco_json = json_files[0]
                try:
                    import json
                    print(f"[Dataset] Parsing COCO annotations from: {coco_json}")
                    with open(coco_json, "r", encoding="utf-8") as f:
                        coco_data = json.load(f)
                    img_id_map = {
                        str(img["id"]): (Path(img["file_name"]).stem, img.get("height", 2048), img.get("width", 2048))
                        for img in coco_data.get("images", [])
                    }
                    for ann in coco_data.get("annotations", []):
                        c_id = str(ann["image_id"])
                        if c_id in img_id_map:
                            stem, h, w = img_id_map[c_id]
                            self.coco_anns.setdefault(stem, []).append({**ann, "_h": h, "_w": w})
                    print(f"[Dataset] Successfully loaded annotations for {len(self.coco_anns)} image stems.")
                except Exception as err:
                    print(f"[Dataset WARNING] Failed to parse COCO JSON: {err}")

        # In-memory preprocessed 4-channel cache
        self._ch4_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

        if self.overfit_single_image:
            self.image_files = [self.image_files[0]]
            print(f"[Dataset] Locked to single overfit image: {self.image_files[0].name} (10 tiles/epoch)")

    def __len__(self) -> int:
        return 10 if self.overfit_single_image else len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = 0 if self.overfit_single_image else (idx % len(self.image_files))
        img_path = self.image_files[real_idx]
        img_id = img_path.stem

        # 1. Retrieve 4-channel image & masks (using offline npy/npz cache or in-memory cache)
        if self.index_meta and img_id in self.index_meta:
            meta = self.index_meta[img_id]
            ch4_img = np.load(meta["npy_path"]).astype(np.float32)
            npz_data = np.load(meta["npz_path"])
            masks_arr = npz_data["masks"]
            h, w = ch4_img.shape[:2]
        elif img_id in self._ch4_cache:
            ch4_img = self._ch4_cache[img_id]
            masks_arr = self._mask_cache[img_id]
            h, w = ch4_img.shape[:2]
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")

            h_img, w_img = img.shape[:2]

            # Load masks from NPZ or COCO JSON
            mask_npz = self.mask_dir / f"{img_id}.npz"
            if mask_npz.exists():
                data = np.load(mask_npz)
                masks_arr = data["masks"]
            elif img_id in self.coco_anns:
                anns = self.coco_anns[img_id]
                masks_list = []
                for ann in anns:
                    seg = ann.get("segmentation")
                    mask = np.zeros((h_img, w_img), dtype=np.uint8)
                    if isinstance(seg, list):
                        for poly in seg:
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], 1)
                        masks_list.append(mask)
                masks_arr = np.stack(masks_list, axis=0) if masks_list else np.zeros((0, h_img, w_img), dtype=np.uint8)
            else:
                masks_arr = np.zeros((0, h_img, w_img), dtype=np.uint8)

            # Apply 4-channel physical preprocessor
            ch4_img = self.preprocessor(img)  # (H, W, 4)
            h, w = ch4_img.shape[:2]

            if self.overfit_single_image:
                self._ch4_cache[img_id] = ch4_img
                self._mask_cache[img_id] = masks_arr

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


def filament_hq_collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Custom collate function handling fixed-size dense tensors and variable instance masks.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    semantics = torch.stack([item["semantic"] for item in batch], dim=0)
    boundaries = torch.stack([item["boundary"] for item in batch], dim=0)
    skeletons = torch.stack([item["skeleton"] for item in batch], dim=0)
    instances = [item["instances"] for item in batch]  # List of (N_i, H, W) tensors
    image_ids = [item["image_id"] for item in batch]

    return {
        "image": images,
        "semantic": semantics,
        "boundary": boundaries,
        "skeleton": skeletons,
        "instances": instances,
        "image_id": image_ids,
    }

