"""
SimMIM Pre-training Script.

Usage:
    python train_simmim.py --data_root <path> --epochs 100 --batch_size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.simmim import SimMIMSegFormer
from src.dataset import SolarFilamentFastDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SimMIM Pre-training")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--backbone", type=str, default="nvidia/mit-b0", help="Backbone model")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size (increased for speed)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs (reduced for Kaggle 12h limit)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (increased for speed)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/simmim", help="Save directory")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency (epochs)")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--stride", type=int, default=512, help="Stride for tiling")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)

        # Generate random mask
        with torch.no_grad():
            mask = model.random_masking(images, model.mask_ratio)

        # Forward pass
        if use_amp:
            with autocast('cuda'):
                outputs = model(images, mask=mask)
                reconstructed = outputs['reconstructed']
                mask = outputs['mask']

                # L1 loss on masked regions only
                loss = F.l1_loss(
                    reconstructed * mask,
                    images * mask,
                    reduction='mean'
                )
        else:
            outputs = model(images, mask=mask)
            reconstructed = outputs['reconstructed']
            mask = outputs['mask']

            loss = F.l1_loss(
                reconstructed * mask,
                images * mask,
                reduction='mean'
            )

        # Backward pass
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
        })

    return {'loss': total_loss / total_samples}


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print("Loading dataset...")
    dataset = SolarFilamentFastDataset(
        data_root=args.data_root,
        tile_size=args.tile_size,
        stride=args.stride,
        fg_ratio=0.0,  # Use all tiles for SimMIM
        bnd_ratio=0.0,
        augment=False,  # No augmentation for SimMIM
        is_train=True,
        in_channels=1,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)} tiles")

    # Model
    print("Creating SimMIM model...")
    model = SimMIMSegFormer(
        backbone_name=args.backbone,
        in_channels=1,
        mask_ratio=args.mask_ratio,
        pretrained=False,  # Random initialization for SimMIM
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler - use PyTorch's built-in schedulers
    # Combined warmup + cosine annealing
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.0,
                end_factor=1.0,
                total_iters=args.warmup_epochs,
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.warmup_epochs,
                eta_min=1e-6,
            ),
        ],
        milestones=[args.warmup_epochs],
    )

    # AMP scaler
    scaler = GradScaler('cuda') if args.use_amp else None

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Total batches per epoch: {len(dataloader)}")
    print(f"Estimated time per epoch: ~{len(dataloader) / 800:.1f} minutes (at 800 it/s)")
    print(f"Estimated total time: ~{len(dataloader) * args.epochs / 800 / 60:.1f} hours")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        metrics = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            args.use_amp,
        )

        print(f"Train Loss: {metrics['loss']:.4f}")

        # Step scheduler after each epoch
        scheduler.step()

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint_path = save_dir / f"simmim_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
                'lr': scheduler.get_last_lr()[0],
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                best_path = save_dir / "simmim_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': metrics['loss'],
                }, best_path)
                print(f"Saved best model: {best_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
