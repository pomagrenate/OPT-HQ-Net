"""
PyTorch Dataset and DataLoader utilities for the MAGFiLO benchmark.

Expected on-disk layout
-----------------------
    data_root/
    ├── images/          # H-Alpha observations (PNG or FITS)
    │   ├── 20150125172714Mh.png
    │   └── …
    ├── masks/           # Per-filament binary masks stored as NPZ
    │   ├── 20150125172714Mh.npz   # key='masks' → (N, H, W) uint8
    │   └── …
    └── boxes.csv        # Optional: pre-computed oriented boxes
                         # columns: image_id, xc, yc, w, h, theta_rad

If only images are present (test split), masks/boxes are skipped.

Usage
-----
>>> from opt_hq_net.data import SolarFilamentDataset, collate_fn
>>> ds = SolarFilamentDataset("data/train", augment=True)
>>> loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from opt_hq_net.data.augmentation import SolarAugmentation
from opt_hq_net.data.preprocessing import CLAHEPreprocessor, SolarDiskMask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset for MAGFiLO solar filament instance segmentation.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing 'images/' and optionally 'masks/'.
    augment : bool
        Whether to apply ``SolarAugmentation``.
    clahe_clip_limit : float
        CLAHE clip limit passed to ``CLAHEPreprocessor``.
    image_extensions : list[str]
        File extensions to recognise as images.
    target_size : int
        All images are resized to (target_size × target_size) before
        being returned.  Use 2048 for full-resolution training.

    Returns (per item)
    ------------------
    dict with keys:
        'image'     — torch.FloatTensor (3, H, W), values in [0, 1]
        'masks'     — torch.BoolTensor  (N, H, W), N instance masks
        'boxes'     — torch.FloatTensor (N, 5)    = [xc, yc, w, h, θ]
        'image_id'  — str, base filename without extension
        'num_instances' — int
    """

    def __init__(
        self,
        data_root: str | Path,
        augment: bool = True,
        clahe_clip_limit: float = 2.0,
        image_extensions: Optional[List[str]] = None,
        target_size: int = 512,
        auto_preprocess: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.augment = augment
        self.target_size = target_size

        # ── Automatic transparent preprocessing check ────────────────────────
        if auto_preprocess:
            is_already_preprocessed = (self.data_root / "masks").exists() and any((self.data_root / "masks").glob("*.npz"))
            if not is_already_preprocessed:
                cache_dir = Path("magfilo_cache") / f"{self.data_root.name}_{target_size}"
                cache_masks = cache_dir / "masks"
                if not cache_masks.exists() or not any(cache_masks.glob("*.npz")):
                    print(f"\n[Dataset] Auto-Preprocessing: Building pre-rendered NPZ cache for '{self.data_root.name}'...")
                    try:
                        preprocess_fn = None
                        try:
                            from opt_hq_net.data.preprocess import preprocess_magfilo_dataset as preprocess_fn
                        except (ImportError, ValueError):
                            try:
                                from .preprocess import preprocess_magfilo_dataset as preprocess_fn
                            except (ImportError, ValueError):
                                try:
                                    from preprocess import preprocess_magfilo_dataset as preprocess_fn
                                except (ImportError, ValueError):
                                    pass

                        if preprocess_fn is not None:
                            preprocess_fn(
                                data_root=self.data_root,
                                output_dir=cache_dir,
                                target_size=target_size,
                                show_progress=True,
                            )
                        else:
                            raise ImportError("Could not locate preprocess_magfilo_dataset module.")
                    except Exception as err:
                        print(f"[Dataset WARNING] Auto-preprocessing skipped: {err}. Falling back to dynamic parsing.")

                if (cache_dir / "images").exists() and any((cache_dir / "images").glob("*.*")):
                    self.data_root = cache_dir

        # Flexible image directory resolution (images/, train_images/, test_images/, or data_root)
        if (self.data_root / "images").exists():
            self.image_dir = self.data_root / "images"
        elif (self.data_root / "train_images").exists():
            self.image_dir = self.data_root / "train_images"
        elif (self.data_root / "test_images").exists():
            self.image_dir = self.data_root / "test_images"
        else:
            self.image_dir = self.data_root

        # Check for COCO JSON annotations or pre-rendered masks/ folder
        self.mask_dir = self.data_root / "masks"
        if self.mask_dir.exists() and any(self.mask_dir.glob("*.npz")):
            self.coco_json = None
        else:
            self.coco_json = self._find_coco_json()

        self.has_masks = (self.coco_json is not None) or (self.mask_dir.exists() and any(self.mask_dir.glob("*.npz")))

        exts = image_extensions or [".png", ".jpg", ".jpeg", ".fits"]
        self.image_ids: List[str] = sorted(
            p.stem
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

        # Load COCO annotations if present
        self.coco_data: Optional[Dict] = None
        self.img_id_to_anns: Dict[str, List[Dict]] = {}
        if self.coco_json is not None:
            self._load_coco_json()

        self._preprocessor = CLAHEPreprocessor(clip_limit=clahe_clip_limit)
        self._disk_masker = SolarDiskMask(margin_px=10)
        self._augmentor = SolarAugmentation(
            min_scale=0.8,
            max_scale=1.2,
            rotation_degrees=360.0,
            crop_sizes=[target_size],
            target_size=target_size,
        ) if augment else None

    def _find_coco_json(self) -> Optional[Path]:
        """Look for COCO format JSON annotations file in data_root or parent directories."""
        # 1. Search in data_root directly and recursively (exclude non-COCO manifest.json)
        json_files = [
            j for j in (list(self.data_root.glob("*.json")) + list(self.data_root.rglob("*.json")))
            if j.name.lower() != "manifest.json"
        ]
        if json_files:
            return json_files[0]

        # 2. Search in parent directories (in case data_root was resolved to train_images/)
        # Filter parent JSON files by matching current split name to prevent train annotations being used for test set
        split_name = self.data_root.name.lower()
        curr = self.data_root.parent
        for _ in range(3):
            if curr and curr.exists():
                parent_jsons = [
                    j for j in (list(curr.glob("*.json")) + list(curr.rglob("*.json")))
                    if j.name.lower() != "manifest.json"
                ]
                matching_jsons = [
                    j for j in parent_jsons
                    if split_name in j.name.lower() or split_name in j.parent.name.lower()
                ]
                if matching_jsons:
                    return matching_jsons[0]
                curr = curr.parent

        return None

    def _load_coco_json(self) -> None:
        """Parse COCO format JSON annotations file."""
        import json
        print(f"[Dataset] Loading COCO annotations from: {self.coco_json}")
        with open(self.coco_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"[Dataset] Skipping non-COCO format JSON file: {self.coco_json}")
            return

        # Build image_filename/stem -> COCO img_id (use str to handle int/str ID types)
        img_id_map = {}
        for img in data.get("images", []):
            file_name = img["file_name"]
            stem = Path(file_name).stem
            img_id_map[str(img["id"])] = (stem, img.get("height", 2048), img.get("width", 2048))

        # Group annotations by image stem
        for ann in data.get("annotations", []):
            coco_img_id = str(ann["image_id"])
            if coco_img_id in img_id_map:
                stem, h, w = img_id_map[coco_img_id]
                if stem not in self.img_id_to_anns:
                    self.img_id_to_anns[stem] = []
                self.img_id_to_anns[stem].append({**ann, "_h": h, "_w": w})

        print(f"[Dataset] Successfully matched annotations for {len(self.img_id_to_anns)} images.")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        image_id = self.image_ids[idx]

        # --- Load image ---
        image_np = self._load_image(image_id)               # (H, W, 3) float32
        image_np = self._disk_masker(image_np)              # zero out background

        # --- Load masks and boxes (train) or create empties (test) ---
        masks_np, boxes_np = self._load_annotations(image_id, image_np.shape[:2])

        # --- Augmentation ---
        if self._augmentor is not None and self.augment:
            image_np, masks_np, boxes_np = self._augmentor(
                image_np, masks_np, boxes_np
            )
        else:
            # Ensure consistent target size even without augmentation
            h, w = image_np.shape[:2]
            if h != self.target_size or w != self.target_size:
                image_np = cv2.resize(image_np, (self.target_size, self.target_size))
                masks_np = SolarAugmentation._resize_masks(
                    masks_np, self.target_size, self.target_size
                )

        # --- Convert to torch tensors ---
        image_t = torch.from_numpy(image_np.transpose(2, 0, 1))          # (3, H, W)
        masks_t = torch.from_numpy(masks_np.astype(np.bool_))            # (N, H, W)
        boxes_t = torch.from_numpy(boxes_np.astype(np.float32))          # (N, 5)

        return {
            "image": image_t,
            "masks": masks_t,
            "boxes": boxes_t,
            "image_id": image_id,
            "num_instances": masks_t.shape[0],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_id: str) -> np.ndarray:
        """Load an image file and apply CLAHE preprocessing."""
        # Try common extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            path = self.image_dir / f"{image_id}{ext}"
            if path.exists():
                raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise IOError(f"Could not read image: {path}")
                return self._preprocessor(raw)

        # FITS fallback (requires astropy)
        fits_path = self.image_dir / f"{image_id}.fits"
        if fits_path.exists():
            try:
                from astropy.io import fits as astropy_fits
                with astropy_fits.open(str(fits_path)) as hdul:
                    raw = hdul[0].data.astype(np.float32)
                return self._preprocessor(raw)
            except ImportError:
                raise ImportError(
                    "astropy is required to load FITS files. "
                    "Install with: pip install astropy"
                )

        raise FileNotFoundError(
            f"No supported image file found for id '{image_id}' in {self.image_dir}"
        )

    def _load_annotations(
        self, image_id: str, image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load instance masks (N, H, W) uint8 and oriented boxes (N, 5).

        If no annotation file exists (test split), returns empty arrays.
        """
        h, w = image_shape

        # Priority 1: Pre-rendered NPZ masks & pre-calculated boxes (Fastest)
        mask_path = self.mask_dir / f"{image_id}.npz"
        if not mask_path.exists():
            mask_path = self.data_root / "masks" / f"{image_id}.npz"
        if mask_path.exists():
            data = np.load(str(mask_path), allow_pickle=False)
            masks = data["masks"].astype(np.uint8)   # (N, H, W)
            if "boxes" in data and len(data["boxes"]) == len(masks):
                boxes = data["boxes"].astype(np.float32)
            else:
                boxes = self._masks_to_oriented_boxes(masks)
            return masks, boxes

        # Priority 2: On-the-fly COCO JSON parsing
        if image_id in self.img_id_to_anns:
            anns = self.img_id_to_anns[image_id]
            masks_list = []
            for ann in anns:
                seg = ann.get("segmentation")
                mask = np.zeros((h, w), dtype=np.uint8)
                if isinstance(seg, list):
                    for poly in seg:
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 1)
                elif isinstance(seg, dict):
                    try:
                        from pycocotools import mask as coco_mask
                        rle = coco_mask.frPyObjects(seg, ann["_h"], ann["_w"])
                        mask = coco_mask.decode(rle).astype(np.uint8)
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    except ImportError:
                        pass
                masks_list.append(mask)

            if masks_list:
                masks = np.stack(masks_list, axis=0)  # (N, H, W)
                boxes = self._masks_to_oriented_boxes(masks)
                return masks, boxes

        return np.zeros((0, h, w), dtype=np.uint8), np.zeros((0, 5), dtype=np.float32)

    @staticmethod
    def _masks_to_oriented_boxes(masks: np.ndarray) -> np.ndarray:
        """
        Fit a minimum-area oriented rectangle to each binary mask.

        Returns (N, 5) array: [xc, yc, w, h, θ_rad].
        The rectangle width is always ≥ height (long axis = width).
        """
        n = masks.shape[0]
        boxes = np.zeros((n, 5), dtype=np.float32)
        for i, mask in enumerate(masks):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            # Merge all contour points
            pts = np.concatenate(contours, axis=0)
            rect = cv2.minAreaRect(pts)          # ((cx, cy), (w, h), angle_deg)
            (cx, cy), (bw, bh), angle_deg = rect
            # Ensure w ≥ h (long axis is w)
            if bw < bh:
                bw, bh = bh, bw
                angle_deg += 90
            theta_rad = math.radians(angle_deg) if bw > 0 else 0.0
            boxes[i] = [cx, cy, bw, bh, theta_rad]
        return boxes


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate for variable-length instance lists.

    Since each image may contain a different number of filament instances,
    we cannot stack tensors along the batch dimension naively.
    Instead, lists are preserved; only the 'image' tensors are stacked.

    Parameters
    ----------
    batch : list[dict]
        Output of ``SolarFilamentDataset.__getitem__``.

    Returns
    -------
    dict with keys:
        'images'      — torch.FloatTensor (B, 3, H, W)
        'masks'       — list[torch.BoolTensor]  length B, each (N_i, H, W)
        'boxes'       — list[torch.FloatTensor] length B, each (N_i, 5)
        'image_ids'   — list[str]
        'num_instances' — list[int]
    """
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": [item["masks"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "num_instances": [item["num_instances"] for item in batch],
    }


# Lazily import math inside the module-level function to avoid circular deps
import math  # noqa: E402  (placed after class definitions intentionally)
