"""
Top-Level CLI Entry Point for Filament-HQ Framework.

Usage Examples:
  # Phase 0: Single-Image FP32 Overfit Test
  python train_hq.py --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/ --overfit_single_image --epochs 100 --imgsz 1024

  # Phase 1: Full Tiled Multi-GPU Training
  python train_hq.py --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/ --imgsz 1024 --batch_size 2 --use_multi_gpu --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from filament_hq import FilamentHQ


def parse_args():
    parser = argparse.ArgumentParser(description="Filament-HQ High-Resolution Training CLI")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone model (default: convnext_tiny)")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Architecture version (v1 or v2)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Fixed input tile size (default: 1024)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Curriculum stage (1, 2, or 3)")
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP FP16 precision")
    parser.add_argument("--overfit_single_image", action="store_true", help="Phase 0: Run single-image FP32 overfit verification test")
    parser.add_argument("--use_multi_gpu", action="store_true", help="Launch DistributedDataParallel across multi-GPUs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_hq", help="Directory to save checkpoints")
    parser.add_argument("--cache", type=str, default=None, help="Path to offline precomputed cache directory")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights for warm-start transfer learning")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n========================================================")
    print("      🚀 FILAMENT-HQ FRAMEWORK INITIALIZED              ")
    print("========================================================")
    print(f" Data Root     : {args.data_root}")
    print(f" Cache Path    : {args.cache or 'None (Online)'}")
    print(f" Pretrained    : {args.weights or 'None (From Scratch)'}")
    print(f" Architecture  : Filament-HQ {args.version.upper()}")
    print(f" Backbone      : {args.backbone}")
    print(f" Native Tile   : {args.imgsz}x{args.imgsz}")
    print(f" Stage         : Stage {args.stage}")
    print(f" AMP Enabled   : {args.use_amp}")
    print(f" Overfit Mode  : {args.overfit_single_image}")
    print("========================================================\n")

    model = FilamentHQ(backbone=args.backbone, version=args.version, weights=args.weights)

    model.train(
        data_root=args.data_root,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stage=args.stage,
        use_amp=args.use_amp,
        overfit_single_image=args.overfit_single_image,
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache,
    )


if __name__ == "__main__":
    main()
