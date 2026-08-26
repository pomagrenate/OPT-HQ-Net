"""
Ultralytics-Style Offline Dataset Cache Builder for Filament-HQ.

Pre-computes:
  1. 4-channel physical representation (C0: raw, C1: contrast, C2: blackhat, C3: radial) -> FP16 .npy
  2. COCO polygon rasterized instance masks -> compressed uint8 .npz
  3. Pre-computed 1024x1024 tile coordinates and foreground density statistics -> index.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add repository root directory to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from filament_hq.data.preprocessor import SolarPhysicalPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics-Style Cache Builder for Filament-HQ")
    parser.add_argument("--data_root", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--output", type=str, default="magfilo_hq_cache", help="Output cache directory")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size (default 1024)")
    parser.add_argument("--stride", type=int, default=768, help="Tile stride for indexing (default 768)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "float16", "float32"],
        help="Precision for cached 4-channel images (uint8 uses ~5.6GB, float16 uses ~23.7GB)",
    )
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="full",
        choices=["full", "morph_only"],
        help="full: cache all 4 channels | morph_only: cache heavy C1 & C2 channels (~2.8GB)",
    )
    return parser.parse_args()


def build_cache():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output)

    img_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate images
    if (data_root / "images").exists():
        img_dir = data_root / "images"
    elif (data_root / "train_images").exists():
        img_dir = data_root / "train_images"
    else:
        img_dir = data_root

    exts = [".png", ".jpg", ".jpeg", ".fits"]
    img_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    print(f"[CacheBuilder] Found {len(img_files)} images in '{img_dir}'.")
    print(f" Cache Mode: {args.cache_mode} | Precision: {args.dtype}")

    # 2. Parse COCO JSON
    coco_anns: Dict[str, List[Dict]] = {}
    json_files = [j for j in (list(data_root.glob("*.json")) + list(data_root.rglob("*.json"))) if j.name.lower() != "manifest.json"]
    if json_files:
        coco_json = json_files[0]
        print(f"[CacheBuilder] Parsing COCO annotations from: {coco_json}")
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
                coco_anns.setdefault(stem, []).append({**ann, "_h": h, "_w": w})

    preprocessor = SolarPhysicalPreprocessor()
    start_time = time.time()

    print(f"[CacheBuilder] Building dataset cache to '{output_dir}'...")

    for img_path in tqdm(img_files, desc="Caching Dataset"):
        stem = img_path.stem
        raw_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            continue

        h_img, w_img = raw_img.shape[:2]

        # 1. 4-Channel Preprocessing
        ch4_img = preprocessor(raw_img)  # (H, W, 4) float32 [0, 1]

        if args.cache_mode == "morph_only":
            ch4_img = ch4_img[:, :, 1:3]  # Keep only C1 (contrast) & C2 (blackhat)

        if args.dtype == "uint8":
            ch4_img = (np.clip(ch4_img, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif args.dtype == "float16":
            ch4_img = ch4_img.astype(np.float16)
        else:
            ch4_img = ch4_img.astype(np.float32)

        np.save(img_out_dir / f"{stem}.npy", ch4_img)

        # 2. Instance Mask Rasterization
        anns = coco_anns.get(stem, [])
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
        np.savez_compressed(mask_out_dir / f"{stem}.npz", masks=masks_arr)

        # 3. Precompute Tile Index & Foreground Density Statistics
        sem_mask = (masks_arr.sum(axis=0) > 0).astype(np.uint8) if len(masks_arr) > 0 else np.zeros((h_img, w_img), dtype=np.uint8)
        tiles_meta = []
        
        for y1 in range(0, h_img - args.tile_size + 1, args.stride):
            for x1 in range(0, w_img - args.tile_size + 1, args.stride):
                tile_sem = sem_mask[y1:y1 + args.tile_size, x1:x1 + args.tile_size]
                fg_ratio = float(tile_sem.mean())
                tiles_meta.append({
                    "y1": y1,
                    "x1": x1,
                    "fg_ratio": fg_ratio,
                    "is_positive": fg_ratio > 0.005,
                })

        index_metadata[stem] = {
            "stem": stem,
            "h": h_img,
            "w": w_img,
            "num_instances": len(masks_arr),
            "npy_path": str(img_out_dir / f"{stem}.npy"),
            "npz_path": str(mask_out_dir / f"{stem}.npz"),
            "tiles": tiles_meta,
        }

    # Save index.pkl
    index_path = output_dir / "index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({
            "version": "1.0.0",
            "tile_size": args.tile_size,
            "stride": args.stride,
            "dtype": args.dtype,
            "items": index_metadata,
        }, f)

    elapsed = time.time() - start_time
    print(f"\n[CacheBuilder] 🎉 Preprocessing complete! Cached {len(index_metadata)} images in {elapsed:.1f}s.")
    print(f" Saved index metadata to: {index_path}\n")


if __name__ == "__main__":
    build_cache()
