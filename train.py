"""
train.py — Entry point for OPT-HQ Net training.

Usage
-----
    python train.py --data_root data/ --epochs 50 --batch_size 2

All hyper-parameters have sensible defaults from ``TrainingConfig``.
Override via CLI arguments.

Example (single GPU)
--------------------
    python train.py \
        --data_root /path/to/magfilo/ \
        --backbone convnext_large \
        --epochs 50 \
        --batch_size 2 \
        --lr 1e-4 \
        --device cuda

Example (quick smoke-test on CPU)
----------------------------------
    python train.py --data_root data/ --epochs 1 --device cpu --batch_size 1
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from opt_hq_net.config import ModelConfig, TrainingConfig
from opt_hq_net.data import SolarFilamentDataset, collate_fn
from opt_hq_net.engine import Trainer
from opt_hq_net.models.opt_hq_net import OPTHQNetBuilder


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OPT-HQ Net on the MAGFiLO solar filament benchmark."
    )
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing 'images/' and 'masks/' subdirs.")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Image size for training crops (default: 512).")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--backbone", type=str, default="convnext_large",
                        choices=["convnext_large", "swin_large"],
                        help="Backbone architecture.")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights.")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")

    # Loss weights
    parser.add_argument("--lambda_box",      type=float, default=1.0)
    parser.add_argument("--lambda_focal",    type=float, default=2.0)
    parser.add_argument("--lambda_dice",     type=float, default=2.0)
    parser.add_argument("--lambda_skeleton", type=float, default=1.5)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Build configs ────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        backbone_name=args.backbone,
        backbone_pretrained=not args.no_pretrained,
    )

    train_cfg = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    train_cfg.loss_weights.oriented_box = args.lambda_box
    train_cfg.loss_weights.focal        = args.lambda_focal
    train_cfg.loss_weights.dice         = args.lambda_dice
    train_cfg.loss_weights.skeleton     = args.lambda_skeleton

    # ── Build model ──────────────────────────────────────────────────────
    print(f"[Main] Building OPT-HQ Net with backbone: {model_cfg.backbone_name}")
    model = OPTHQNetBuilder(model_cfg).build()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Trainable parameters: {n_params / 1e6:.1f}M")

    # ── Build datasets ───────────────────────────────────────────────────
    data_root_path = Path(args.data_root)
    if (data_root_path / "train").exists():
        train_path = data_root_path / "train"
    else:
        train_path = data_root_path

    if (data_root_path / "val").exists():
        val_path = data_root_path / "val"
    elif (data_root_path / "test").exists():
        val_path = data_root_path / "test"
    else:
        val_path = train_path

    print(f"[Main] Loading train dataset from: {train_path}")
    print(f"[Main] Loading val dataset from:   {val_path}")

    train_ds = SolarFilamentDataset(
        data_root=train_path,
        augment=True,
        target_size=args.target_size,
    )
    val_ds = SolarFilamentDataset(
        data_root=val_path,
        augment=False,
        target_size=args.target_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
    )

    print(f"[Main] Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, train_cfg)
    trainer.train()


if __name__ == "__main__":
    main()
