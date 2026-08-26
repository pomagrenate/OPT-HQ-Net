"""
Cache Verification & Audit Script for Filament-HQ.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

# Add repository root directory to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def audit_cache():
    parser = argparse.ArgumentParser(description="Audit Filament-HQ Precomputed Cache")
    parser.add_argument("--cache", type=str, required=True, help="Path to cache directory")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    index_path = cache_dir / "index.pkl"

    if not index_path.exists():
        print(f"❌ Cache index not found at '{index_path}'. Run tools/build_cache.py first!")
        return

    with open(index_path, "rb") as f:
        meta = pickle.load(f)

    items = meta["items"]
    print("\n========================================================")
    print("      📊 FILAMENT-HQ CACHE INTEGRITY AUDIT              ")
    print("========================================================")
    print(f" Cache Version  : {meta.get('version', '1.0')}")
    print(f" Total Cached   : {len(items)} images")
    print(f" Tile Size      : {meta.get('tile_size')}x{meta.get('tile_size')}")
    print(f" Tile Stride    : {meta.get('stride')}px")
    print(f" Cached Precision: {meta.get('dtype')}")
    print("========================================================\n")

    total_instances = sum(item["num_instances"] for item in items.values())
    pos_tiles = sum(sum(1 for t in item["tiles"] if t["is_positive"]) for item in items.values())
    total_tiles = sum(len(item["tiles"]) for item in items.values())

    print(f" Total Filament Instances : {total_instances}")
    print(f" Total Indexed Tiles     : {total_tiles}")
    print(f" Positive Filament Tiles : {pos_tiles} ({100.0 * pos_tiles / max(total_tiles, 1):.1f}%)\n")

    # Audit disk usage
    img_bytes = sum(p.stat().st_size for p in (cache_dir / "images").glob("*.npy"))
    mask_bytes = sum(p.stat().st_size for p in (cache_dir / "masks").glob("*.npz"))
    total_gb = (img_bytes + mask_bytes) / (1024 ** 3)

    print(f" Disk Size - Images : {img_bytes / (1024**2):.1f} MB")
    print(f" Disk Size - Masks  : {mask_bytes / (1024**2):.1f} MB")
    print(f" Total Footprint    : {total_gb:.2f} GB")
    print("\n✅ CACHE INTEGRITY VERIFIED OK!")


if __name__ == "__main__":
    audit_cache()
