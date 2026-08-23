"""
predict.py — Entry point for OPT-HQ Net inference and submission generation.

Usage
-----
    python predict.py \
        --data_root data/test \
        --checkpoint checkpoints/opt_hq_net_best_0050.pth \
        --output submission.csv \
        --device cuda

The script produces a submission.csv ready for upload to the MAGFiLO
benchmark leaderboard.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from opt_hq_net.config import ModelConfig
from opt_hq_net.data import SolarFilamentDataset, collate_fn
from opt_hq_net.engine import InferencePipeline, Trainer
from opt_hq_net.models.opt_hq_net import OPTHQNetBuilder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OPT-HQ Net inference and generate submission CSV."
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing 'images/' (test split).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth).")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output CSV file path.")
    parser.add_argument("--backbone", type=str, default="convnext_large")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--nms_iou", type=float, default=0.40)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--min_area", type=int, default=50)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Model ────────────────────────────────────────────────────────────
    print(f"[Predict] Loading model from: {args.checkpoint}")
    model_cfg = ModelConfig(backbone_name=args.backbone, backbone_pretrained=False)
    model = OPTHQNetBuilder(model_cfg).build()
    model = Trainer.load_checkpoint(model, args.checkpoint, device=args.device)

    # ── Dataset ──────────────────────────────────────────────────────────
    from pathlib import Path
    data_root_path = Path(args.data_root)
    if (data_root_path / "test").exists():
        test_path = data_root_path / "test"
    else:
        test_path = data_root_path

    print(f"[Predict] Loading test dataset from: {test_path}")

    test_ds = SolarFilamentDataset(
        data_root=test_path,
        augment=False,
        target_size=args.target_size,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    print(f"[Predict] Test images: {len(test_ds)}")

    # ── Inference ────────────────────────────────────────────────────────
    pipeline = InferencePipeline(
        model=model,
        device=args.device,
        nms_iou_threshold=args.nms_iou,
        score_threshold=args.score_threshold,
        min_mask_area_px=args.min_area,
    )

    predictions = pipeline.run_on_dataset(test_loader)

    total_instances = sum(len(v) for v in predictions.values())
    print(f"[Predict] Total filament instances detected: {total_instances}")

    # ── CSV ──────────────────────────────────────────────────────────────
    submission_df = pipeline.build_submission(predictions)
    submission_df.to_csv(args.output, index=False)
    print(f"[Predict] Submission saved to: {args.output}")
    print(f"[Predict] Rows: {len(submission_df)}")


if __name__ == "__main__":
    main()
