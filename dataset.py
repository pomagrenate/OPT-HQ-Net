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
    preprocess_halpha,
)


class SolarFilamentDataset(Dataset):
    SUPPORTED_EXTENSIONS = ('.npy', '.fits', '.fit', '.jpeg', '.jpg', '.png')

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        use_cache: bool = True,
        use_mmap: bool = True,
        augment: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache_limit: int = 8,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.use_cache = use_cache
        self.use_mmap = use_mmap
        self.augment = augment and (self.split == 'train')
        self.transform = transform
        self.cache_limit = max(0, cache_limit)

        self.image_dir = self._resolve_image_dir()
        self.image_files = self._collect_image_files()
        if not self.image_files:
            raise FileNotFoundError(f"No valid image files found in {self.image_dir}")

        self.img_to_polygons: Dict[str, List[List[float]]] = {}
        self.img_dimensions: Dict[str, Tuple[int, int]] = {}
        self._cache_store: Dict[str, Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]] = {}
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
            return np.asarray(arr, dtype=np.float32)

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

    def _get_processed_data(self, path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        key = str(path)
        if key in self._cache_store:
            return self._cache_store[key]

        raw_arr = self._read_image(path)
        if raw_arr.ndim == 3 and raw_arr.shape[0] == 1:
            raw_arr = raw_arr[0]

        if raw_arr.ndim == 2:
            clean_img, mask, meta = preprocess_halpha(raw_arr)
        elif raw_arr.ndim == 3:
            clean_img = raw_arr[0]
            mask = np.ones_like(clean_img)
            h, w = clean_img.shape
            meta = (w // 2, h // 2, int(0.46 * min(h, w)))
        else:
            raise ValueError(f"Unexpected image shape {raw_arr.shape} at {path}")

        res = (clean_img, mask, meta)
        if len(self._cache_store) < self.cache_limit:
            self._cache_store[key] = res
        return res

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        clean_img, valid_mask, (cx, cy, r_sun) = self._get_processed_data(img_path)
        h, w = clean_img.shape

        img_full = clean_img[np.newaxis, ...]
        valid_full = valid_mask[np.newaxis, ...]

        if self.split == 'train':
            gt_mask = self._generate_mask(img_path.name, fallback_shape=(h, w))
            if gt_mask is None:
                gt_mask = np.zeros((h, w), dtype=np.float32)

            if self.augment:
                img_full, gt_mask, valid_full = continuity_safe_augment(
                    img_full, gt_mask, valid_full
                )

            sample: Dict[str, Any] = {
                'image': torch.from_numpy(np.ascontiguousarray(img_full)).float(),
                'valid_mask': torch.from_numpy(np.ascontiguousarray(valid_full)).float(),
                'mask': torch.from_numpy(np.ascontiguousarray(gt_mask)).float().unsqueeze(0),
                'has_filament': bool(gt_mask.sum() > 0),
                'image_id': img_path.stem,
            }
        else:
            sample = {
                'image': torch.from_numpy(np.ascontiguousarray(img_full)).float(),
                'valid_mask': torch.from_numpy(np.ascontiguousarray(valid_full)).float(),
                'mask': None,
                'image_id': img_path.stem,
                'disk': (cx, cy, r_sun),
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
            'valid_mask': torch.stack([b['valid_mask'] for b in batch], dim=0),
            'mask': torch.stack([b['mask'] for b in batch], dim=0),
            'has_filament': torch.tensor([b['has_filament'] for b in batch], dtype=torch.bool),
            'image_id': [b['image_id'] for b in batch],
        }

    return {
        'image': torch.stack([b['image'] for b in batch], dim=0),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch], dim=0),
        'image_id': [b['image_id'] for b in batch],
        'disk': [b.get('disk') for b in batch],
    }


def create_dataloaders(
    data_root: str | Path,
    batch_size: int = 1,
    val_split: float = 0.1,
    num_workers: int = 2,
    use_cache: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    full_dataset = SolarFilamentDataset(
        data_root=data_root,
        split='train',
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