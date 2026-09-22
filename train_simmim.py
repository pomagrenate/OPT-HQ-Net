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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.simmim import SimMIMSegFormer
from src.dataset import SolarFilamentFastDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SimMIM Pre-training")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--backbone", type=str, default="nvidia/mit-b0", help="Backbone model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints/simmim", help="Save directory")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency (epochs)")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--stride", type=int, default=512, help="Stride for tiling")
    return parser.parse_args()


class CosineAnnealingWithWarmup:
    """Cosine annealing scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self) -> None:
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingWithWarmup,
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
            with autocast():
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

        scheduler.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{scheduler.get_lr():.6f}"
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

    # Scheduler
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
    )

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_lr():.6f}")

        metrics = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            scaler,
            device,
            args.use_amp,
        )

        print(f"Train Loss: {metrics['loss']:.4f}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint_path = save_dir / f"simmim_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
                'lr': scheduler.get_lr(),
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
