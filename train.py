"""
train.py — Primary CLI Entrypoint for Ultra-Efficient Solar Filament Micro-Segmentation.

Supports:
  - Kaggle 2x GPU DistributedDataParallel (DDP) and clean single-GPU execution.
  - Automatic Mixed Precision (AMP FP16).
  - Resuming broken or interrupted training: --resume checkpoints/last.pt
  - 512x512 tile micro-segmentation with zero CPU bottlenecks.

Usage Examples:
  # Kaggle Single GPU / Local:
  python train.py --data_root /path/to/MAGFiLO --batch_size 4 --epochs 50 --use_amp

  # Kaggle 2x GPU with torchrun (Recommended):
  torchrun --nproc_per_node=2 train.py --data_root /path/to/MAGFiLO --batch_size 4 --epochs 50 --use_amp

  # Resume training after timeout or interruption:
  python train.py --data_root /path/to/MAGFiLO --resume checkpoints/last.pt --epochs 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src import SolarFilamentFastDataset, SolarFilamentNet, SolarTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solar Filament Fast Micro-Segmentation Trainer")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--backbone", type=str, default="resnet34", help="timm backbone model (default: resnet34)")
    parser.add_argument("--tile_size", type=int, default=512, help="Patch resolution (default: 512)")
    parser.add_argument("--stride", type=int, default=384, help="Patch stride (default: 384)")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable AMP FP16 precision (default: True)")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp", help="Disable AMP")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from, or 'last'")
    parser.add_argument("--save_interval", type=int, default=1, help="Interval in epochs to save milestone checkpoints")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes per GPU (default: 2)")
    return parser.parse_args()


def init_ddp() -> Tuple[int, int, bool]:
    """Initialize torch.distributed if running in DDP environment (e.g. torchrun)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return rank, world_size, True
    return 0, 1, False


def main() -> None:
    args = parse_args()
    rank, world_size, is_ddp = init_ddp()
    is_master = (rank == 0)

    if is_master:
        print("\n========================================================")
        print("   SOLAR FILAMENT MICRO-SEGMENTATION FRAMEWORK (v2.0)   ")
        print("========================================================")
        print(f" Data Root     : {args.data_root}")
        print(f" Backbone      : {args.backbone}")
        print(f" Tile Resolution: {args.tile_size}x{args.tile_size} (Stride: {args.stride})")
        print(f" Batch Size    : {args.batch_size} per GPU (Total: {args.batch_size * world_size})")
        print(f" Target Epochs : {args.epochs}")
        print(f" AMP Enabled   : {args.use_amp}")
        print(f" Resuming From : {args.resume or 'None (Training from Scratch)'}")
        print(f" DDP Active    : {is_ddp} (World Size: {world_size})")
        print(f" Checkpoint Dir: {args.checkpoint_dir}")
        print("========================================================\n")

    # 1. Dataset & DataLoaders
    train_dataset = SolarFilamentFastDataset(
        data_root=args.data_root,
        tile_size=args.tile_size,
        stride=args.stride,
        fg_ratio=0.70,
        bnd_ratio=0.20,
        augment=True,
        is_train=True,
    )

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    # Validation DataLoader (only master process evaluates to prevent duplicate compute)
    val_loader = None
    if is_master:
        val_dataset = SolarFilamentFastDataset(
            data_root=args.data_root,
            tile_size=args.tile_size,
            stride=args.stride,
            augment=False,
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # 2. Model & Optimizer
    model = SolarFilamentNet(
        backbone_name=args.backbone,
        in_channels=3,
        decoder_channels=128,
        pretrained=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. Trainer
    trainer = SolarTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        use_amp=args.use_amp,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        save_interval=args.save_interval,
        rank=rank,
        world_size=world_size,
    )

    trainer.train()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
