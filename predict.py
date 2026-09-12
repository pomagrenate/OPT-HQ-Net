"""
predict.py — High-Throughput Inference & Kaggle Submission Generator.

Usage:
  python predict.py \
    --weights checkpoints/best_model.pt \
    --data_root filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test \
    --output submission.csv \
    --save_masks output_masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import FastPatchInferer, SolarFilamentNet, binary_mask_to_rle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar Filament Fast Inference & Kaggle Submission Generator")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path for output submission CSV")
    parser.add_argument("--save_masks", type=str, default=None, help="Optional directory to save predicted mask PNGs")
    parser.add_argument("--backbone", type=str, default="resnet34", help="Encoder backbone name (default: resnet34)")
    parser.add_argument("--tile_size", type=int, default=512, help="Patch resolution (default: 512)")
    parser.add_argument("--stride", type=int, default=384, help="Patch stride (default: 384)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Binarization probability threshold (default: 0.50)")
    parser.add_argument("--min_area", type=int, default=50, help="Minimum pixel area for connected filament instances")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n========================================================")
    print("   SOLAR FILAMENT HIGH-THROUGHPUT INFERENCE PIPELINE    ")
    print("========================================================")
    print(f" Weights       : {args.weights}")
    print(f" Test Data     : {args.data_root}")
    print(f" Device        : {device}")
    print(f" Threshold     : {args.threshold}")
    print(f" Output CSV    : {args.output}")
    print("========================================================\n")

    # 1. Build Model & Load Weights
    model = SolarFilamentNet(backbone_name=args.backbone, in_channels=3, decoder_channels=128, pretrained=False)
    ckpt = torch.load(args.weights, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format in '{args.weights}'")

    model.load_state_dict(state_dict)
    model.to(device).eval()

    inferer = FastPatchInferer(
        model=model,
        tile_size=args.tile_size,
        stride=args.stride,
        device=device,
        batch_size=25,
    )

    # 2. Discover Test Images
    test_dir = Path(args.data_root)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    img_files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in valid_exts])
    if not img_files:
        raise FileNotFoundError(f"No image files found in '{args.data_root}'")

    print(f"Found {len(img_files)} test images to process.")

    if args.save_masks:
        out_mask_dir = Path(args.save_masks)
        out_mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_mask_dir = None

    # 3. Process Images & Generate RLE Submissions
    records = []

    for img_path in tqdm(img_files, desc="Running Inference"):
        base_id = img_path.stem
        res = inferer.predict(img_path, threshold=args.threshold, use_amp=True)
        bin_mask = res["binary_mask"]

        if out_mask_dir is not None:
            cv2.imwrite(str(out_mask_dir / f"{base_id}_pred.png"), bin_mask * 255)

        # Instance Separation via connected components
        num_labels, labels = cv2.connectedComponents(bin_mask)

        inst_count = 0
        for label_idx in range(1, num_labels):
            inst_mask = (labels == label_idx).astype(np.uint8)
            if inst_mask.sum() >= args.min_area:
                inst_count += 1
                rle_str = binary_mask_to_rle(inst_mask)
                records.append({
                    "filament_id": f"{base_id}_{inst_count}",
                    "segmentation_rle": rle_str,
                })

        # If no instances met the threshold, append empty record to guarantee coverage
        if inst_count == 0:
            records.append({
                "filament_id": f"{base_id}_1",
                "segmentation_rle": "",
            })

    # 4. Save CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"\n[Success] Generated submission with {len(df)} predictions saved to '{args.output}'")


if __name__ == "__main__":
    main()
