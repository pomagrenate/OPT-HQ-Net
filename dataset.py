from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

try:
    from astropy.io import fits
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False

from preprocessing import (
    continuity_safe_augment,
    process_solar_observation,
)


class SolarFilamentDataset(Dataset):
    SUPPORTED_EXTENSIONS = ('.npy', '.fits', '.fit', '.jpeg', '.jpg', '.png')

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        tile_size: int = 512,
        overlap: float = 0.25,
        use_cache: bool = True,
        use_mmap: bool = True,
        augment: bool = False,
        global_size: int = 512,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_cache = use_cache
        self.use_mmap = use_mmap
        self.augment = augment and (self.split == 'train')
        self.global_size = global_size
        self.transform = transform

        self.image_dir = self._resolve_image_dir()
        self.image_files = self._collect_image_files()
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in {self.image_dir}")

        self.img_to_polygons: Dict[str, List[List[float]]] = {}
        self.img_dimensions: Dict[str, Tuple[int, int]] = {}
        if self.split == 'train':
            self._load_and_index_annotations()

    def _resolve_image_dir(self) -> Path:
        sub = "train" if self.split == "train" else "test"
        candidate_paths = [
            self.data_root / sub / f"{sub}_images",
            self.data_root / f"{sub}_images",
            self.data_root / sub,
            self.data_root,
        ]
        for path in candidate_paths:
            if path.is_dir():
                return path
        raise FileNotFoundError(
            f"Could not locate image directory for split '{self.split}'. Checked: {candidate_paths}"
        )

    def _collect_image_files(self) -> List[Path]:
        if self.use_cache:
            npy_files = sorted(list(self.image_dir.glob("*.npy")))
            if npy_files:
                return npy_files

        files: List[Path] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            if ext == '.npy':
                continue
            files.extend(self.image_dir.glob(f"*{ext}"))
            files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return sorted(files)

    def _load_and_index_annotations(self) -> None:
        candidate_files = [
            self.data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
            self.data_root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
            self.data_root / "train" / "annotations.json",
            self.data_root / "annotations.json",
        ]
        ann_path = next((p for p in candidate_files if p.is_file()), None)
        if ann_path is None:
            return

        with open(ann_path, "r", encoding="utf-8") as f:
            coco_payload = json.load(f)

        id_to_filename: Dict[int, str] = {}
        for img_info in coco_payload.get("images", []):
            img_id = img_info["id"]
            fname = img_info["file_name"]
            id_to_filename[img_id] = fname
            self.img_dimensions[fname] = (
                img_info.get("height", 2048),
                img_info.get("width", 2048),
            )

        for ann in coco_payload.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in id_to_filename:
                continue

            fname = id_to_filename[img_id]
            if fname not in self.img_to_polygons:
                self.img_to_polygons[fname] = []

            segmentation = ann.get("segmentation", [])
            if isinstance(segmentation, list):
                self.img_to_polygons[fname].extend(segmentation)

    def _read_image(self, path: Path) -> np.ndarray:
        ext = path.suffix.lower()

        if ext == '.npy':
            mmap = 'r' if self.use_mmap else None
            arr = np.load(path, mmap_mode=mmap)
            return np.array(arr, dtype=np.float32, copy=False)

        if ext in ('.fits', '.fit'):
            if not _HAS_ASTROPY:
                raise ImportError("astropy library is required to read FITS files.")
            with fits.open(path) as hdul:
                arr = hdul[0].data.astype(np.float32)
        else:
            with Image.open(path) as img:
                arr = np.array(img.convert('L'), dtype=np.float32)

        max_val = arr.max()
        if max_val > 1.0:
            arr /= 255.0 if max_val <= 255.0 else max_val

        return arr

    def _generate_mask(self, file_name: str, fallback_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        polygons = self.img_to_polygons.get(file_name)
        if polygons is None:
            polygons = next(
                (v for k, v in self.img_to_polygons.items() if Path(k).stem == Path(file_name).stem),
                None,
            )

        if not polygons:
            return None

        h, w = self.img_dimensions.get(file_name, fallback_shape)
        mask = np.zeros((h, w), dtype=np.uint8)

        for poly in polygons:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], color=1)

        return mask.astype(np.float32)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        raw_arr = self._read_image(img_path)

        if raw_arr.ndim == 3 and raw_arr.shape[0] == 2:
            image_stack = raw_arr
            h, w = image_stack.shape[1], image_stack.shape[2]
        elif raw_arr.ndim == 2:
            h, w = raw_arr.shape
            image_stack = process_solar_observation(raw_arr)
        else:
            raise ValueError(f"Unexpected image shape {raw_arr.shape} at {img_path}")

        cx, cy = w // 2, h // 2
        r_sun = int(0.46 * min(h, w))

        global_img = cv2.resize(image_stack[0], (self.global_size, self.global_size), interpolation=cv2.INTER_AREA)
        global_ridge = cv2.resize(image_stack[1], (self.global_size, self.global_size), interpolation=cv2.INTER_AREA)
        global_stack = np.stack([global_img, global_ridge], axis=0).astype(np.float32)

        valid_mask = np.ones((1, h, w), dtype=np.float32)

        if self.split == 'train':
            gt_mask = self._generate_mask(img_path.name, fallback_shape=(h, w))
            if gt_mask is None:
                gt_mask = np.zeros((h, w), dtype=np.float32)

            polygons = self.img_to_polygons.get(img_path.name, [])
            sample_positive = (len(polygons) > 0) and (np.random.rand() < 0.75)

            if sample_positive and len(polygons) > 0:
                chosen_poly = polygons[np.random.randint(len(polygons))]
                poly_pts = np.array(chosen_poly).reshape(-1, 2)
                target_x = int(poly_pts[:, 0].mean())
                target_y = int(poly_pts[:, 1].mean())

                margin = self.tile_size // 4
                tx = target_x - self.tile_size // 2 + np.random.randint(-margin, margin + 1)
                ty = target_y - self.tile_size // 2 + np.random.randint(-margin, margin + 1)
            else:
                tx = np.random.randint(0, max(1, w - self.tile_size + 1))
                ty = np.random.randint(0, max(1, h - self.tile_size + 1))

            tx = int(np.clip(tx, 0, max(0, w - self.tile_size)))
            ty = int(np.clip(ty, 0, max(0, h - self.tile_size)))

            img_tile = image_stack[:, ty : ty + self.tile_size, tx : tx + self.tile_size]
            valid_tile = valid_mask[:, ty : ty + self.tile_size, tx : tx + self.tile_size]
            gt_tile = gt_mask[ty : ty + self.tile_size, tx : tx + self.tile_size]
            has_fil = bool(gt_tile.sum() > 0)

            if self.augment:
                img_tile, gt_tile, valid_tile = continuity_safe_augment(
                    img_tile, gt_tile, valid_tile
                )

            center_y = ty + self.tile_size / 2.0
            center_x = tx + self.tile_size / 2.0
            x_norm = center_x / float(w)
            y_norm = center_y / float(h)
            dist_sun = np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2)
            r_norm = dist_sun / max(float(r_sun), 1.0)
            coords = np.array([x_norm, y_norm, r_norm], dtype=np.float32)

            sample: Dict[str, Any] = {
                'image': torch.from_numpy(np.ascontiguousarray(img_tile)).float(),
                'global_image': torch.from_numpy(np.ascontiguousarray(global_stack)).float(),
                'coords': torch.from_numpy(coords).float(),
                'valid_mask': torch.from_numpy(np.ascontiguousarray(valid_tile)).float(),
                'mask': torch.from_numpy(np.ascontiguousarray(gt_tile)).float().unsqueeze(0),
                'has_filament': has_fil,
                'tile_coords': torch.tensor([ty, tx], dtype=torch.long),
                'image_id': img_path.stem,
            }
        else:
            disk_meta = (cx, cy, r_sun)
            sample = {
                'image': torch.from_numpy(np.ascontiguousarray(image_stack)).float(),
                'global_image': torch.from_numpy(np.ascontiguousarray(global_stack)).float(),
                'valid_mask': torch.from_numpy(np.ascontiguousarray(valid_mask)).float(),
                'mask': None,
                'image_id': img_path.stem,
                'disk': disk_meta,
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def solar_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    has_mask = batch[0].get('mask') is not None

    if has_mask:
        return {
            'image': torch.stack([b['image'] for b in batch], dim=0),
            'global_image': torch.stack([b['global_image'] for b in batch], dim=0),
            'coords': torch.stack([b['coords'] for b in batch], dim=0),
            'valid_mask': torch.stack([b['valid_mask'] for b in batch], dim=0),
            'mask': torch.stack([b['mask'] for b in batch], dim=0),
            'has_filament': torch.tensor([b['has_filament'] for b in batch], dtype=torch.bool),
            'tile_coords': torch.stack([b['tile_coords'] for b in batch], dim=0),
            'image_id': [b['image_id'] for b in batch],
        }

    return {
        'image': [b['image'] for b in batch],
        'global_image': [b['global_image'] for b in batch],
        'valid_mask': [b['valid_mask'] for b in batch],
        'image_id': [b['image_id'] for b in batch],
        'disk': [b.get('disk') for b in batch],
    }


def create_dataloaders(
    data_root: str | Path,
    batch_size: int = 4,
    tile_size: int = 512,
    overlap: float = 0.25,
    val_split: float = 0.1,
    num_workers: int = 2,
    use_cache: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    full_dataset = SolarFilamentDataset(
        data_root=data_root,
        split='train',
        tile_size=tile_size,
        overlap=overlap,
        use_cache=use_cache,
        augment=True,
    )

    total_samples = len(full_dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=solar_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=solar_collate_fn,
        drop_last=False,
    ) if val_size > 0 else None

    return train_loader, val_loader