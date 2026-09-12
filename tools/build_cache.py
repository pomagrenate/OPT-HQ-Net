"""
Simple Offline Dataset Cache Builder for Solar Filament Segmentation.

Converts raw H-alpha images and COCO polygon annotations into lightweight
NumPy (.npy) arrays for ultra-fast disk I/O and zero-CPU rasterization training.

Usage:
    python tools/build_cache.py \
        --data_root /path/to/MAGFiLO_1.0_Kaggle_2026/train \
        --output /path/to/magfilo_hq_cache
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Solar Filament Dataset to .npy Cache")
    parser.add_argument("--data_root", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--output", type=str, default="magfilo_hq_cache", help="Output cache directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max images to cache (for testing)")
    
    # Compatibility arguments (ignored to prevent CLI breakage if passed by user)
    parser.add_argument("--dtype", type=str, default="uint8", help="Data type (default uint8, kept for compatibility)")
    parser.add_argument("--cache_mode", type=str, default="full", help="Cache mode (kept for compatibility)")
    
    return parser.parse_args()


def build_cache():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output)

    if not data_root.exists():
        print(f"[ERROR] data_root '{data_root}' does not exist!")
        sys.exit(1)

    img_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate raw images directory
    img_dir = None
    for cand in ["train_images", "images", "train"]:
        if (data_root / cand).is_dir():
            img_dir = data_root / cand
            break
    if img_dir is None:
        img_dir = data_root

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".fits"}
    img_files = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not img_files:
        img_files = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    print(f"[CacheBuilder] Found {len(img_files)} raw images in '{img_dir}'.")
    if args.max_samples is not None:
        img_files = img_files[: args.max_samples]
        print(f"[CacheBuilder] Capping to {len(img_files)} samples (--max_samples).")

    # 2. Parse COCO JSON annotations if available
    coco_anns: Dict[str, List[np.ndarray]] = {}
    json_files = [j for j in data_root.rglob("*.json") if "manifest" not in j.name.lower()]
    source_json = None

    if json_files:
        source_json = json_files[0]
        print(f"[CacheBuilder] Parsing COCO annotations from: {source_json.name}")
        try:
            with open(source_json, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

            img_id_to_stem = {}
            for img_info in coco_data.get("images", []):
                fname = img_info.get("file_name", "")
                stem = Path(fname).stem
                img_id_to_stem[img_info["id"]] = stem

            for ann in coco_data.get("annotations", []):
                i_id = ann.get("image_id")
                if i_id in img_id_to_stem:
                    stem = img_id_to_stem[i_id]
                    seg = ann.get("segmentation", [])
                    if isinstance(seg, list):
                        for poly in seg:
                            if len(poly) >= 6:
                                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                                coco_anns.setdefault(stem, []).append(pts)

            print(f"[CacheBuilder] Indexed annotations for {len(coco_anns)} images.")
        except Exception as e:
            print(f"[CacheBuilder] Warning: Failed to parse COCO annotations: {e}")

    # 3. Process & save .npy files
    start_time = time.time()
    num_cached = 0
    total_imgs = len(img_files)

    print(f"[CacheBuilder] Converting images and rasterizing masks to .npy in '{output_dir}'...")

    for i, img_path in enumerate(img_files):
        stem = img_path.stem
        raw_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raw_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if raw_img is None:
            print(f"[WARN] Unable to load '{img_path}', skipping.")
            continue

        h, w = raw_img.shape[:2]

        # Optimize storage: If all 3 channels are identical (MAGFiLO H-alpha standard),
        # store as 2D uint8 (2048, 2048) to reduce Kaggle disk usage from ~14GB to ~4.7GB
        if raw_img.ndim == 3 and raw_img.shape[2] == 3:
            if np.array_equal(raw_img[:, :, 0], raw_img[:, :, 1]) and np.array_equal(raw_img[:, :, 0], raw_img[:, :, 2]):
                save_img = raw_img[:, :, 0].astype(np.uint8)
            else:
                save_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        else:
            save_img = raw_img.astype(np.uint8)

        # Save image as .npy
        np.save(img_out_dir / f"{stem}.npy", save_img)

        # Generate & save mask as .npy
        mask = np.zeros((h, w), dtype=np.uint8)
        if stem in coco_anns:
            for pts in coco_anns[stem]:
                cv2.fillPoly(mask, [pts], 1)
        else:
            # Check for existing mask file fallback
            mask_fallback = data_root / "masks" / f"{stem}.png"
            if mask_fallback.exists():
                loaded_m = cv2.imread(str(mask_fallback), cv2.IMREAD_GRAYSCALE)
                if loaded_m is not None:
                    mask = (loaded_m > 127).astype(np.uint8)

        np.save(mask_out_dir / f"{stem}.npy", mask)
        num_cached += 1

        if (i + 1) % 100 == 0 or (i + 1) == total_imgs:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(0.01, elapsed)
            print(f"  [{i + 1:04d}/{total_imgs:04d}] Cached -> {stem}.npy ({rate:.1f} imgs/s)")

    # 4. Copy/link COCO JSON annotations if available
    if source_json and source_json.exists():
        dest_json = output_dir / source_json.name
        try:
            shutil.copy2(source_json, dest_json)
            # Also provide standard annotations.json name
            shutil.copy2(source_json, output_dir / "annotations.json")
            print(f"[CacheBuilder] Copied annotations to: {output_dir / 'annotations.json'}")
        except Exception as e:
            print(f"[CacheBuilder] Note: Could not copy JSON annotations: {e}")

    total_time = time.time() - start_time
    print("\n========================================================")
    print(f"[CacheBuilder] Caching Complete! Successfully converted {num_cached} images to .npy")
    print(f"  Images directory: {img_out_dir}")
    print(f"  Masks directory : {mask_out_dir}")
    print(f"  Total time      : {total_time:.2f}s ({num_cached / max(0.01, total_time):.1f} imgs/s)")
    print("========================================================\n")


if __name__ == "__main__":
    build_cache()
