"""
Supervised Segmentation Fine-tuning Script with SimMIM Pre-training.

Usage:
    python train_segmentation.py --data_root <path> --simmim_checkpoint <path> --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.segmentation import SolarFilamentSegmentation
from src.dataset import SolarFilamentFastDataset
from src.losses import CompoundLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Segmentation Fine-tuning")
    parser.add_argument("--data_root", type=str, required=True, help="Path to training data")
    parser.add_argument("--backbone", type=str, default="nvidia/mit-b0", help="Backbone model")
    parser.add_argument("--simmim_checkpoint", type=str, default=None, help="Path to SimMIM checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints/segmentation", help="Save directory")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency (epochs)")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--tile_size", type=int, default=512, help="Tile size")
    parser.add_argument("--stride", type=int, default=512, help="Stride for tiling")
    parser.add_argument("--fg_ratio", type=float, default=0.70, help="Foreground tile ratio")
    parser.add_argument("--bnd_ratio", type=float, default=0.20, help="Boundary tile ratio")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_cldice = 0.0
    total_skel = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        if use_amp:
            with autocast('cuda'):
                logits = model(images)
                loss_dict = criterion(logits, masks)
                loss = loss_dict['loss']
        else:
            logits = model(images)
            loss_dict = criterion(logits, masks)
            loss = loss_dict['loss']

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
        total_bce += loss_dict['loss_bce'].item() * images.size(0)
        total_dice += loss_dict['loss_dice'].item() * images.size(0)
        total_cldice += loss_dict['loss_cldice'].item() * images.size(0)
        total_skel += loss_dict['loss_skel'].item() * images.size(0)
        total_samples += images.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{loss_dict['loss_dice'].item():.4f}",
            'cldice': f"{loss_dict['loss_cldice'].item():.4f}",
        })

    return {
        'loss': total_loss / total_samples,
        'loss_bce': total_bce / total_samples,
        'loss_dice': total_dice / total_samples,
        'loss_cldice': total_cldice / total_samples,
        'loss_skel': total_skel / total_samples,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss_dict = criterion(logits, masks)

            total_loss += loss_dict['loss'].item() * images.size(0)
            total_dice += loss_dict['loss_dice'].item() * images.size(0)
            total_samples += images.size(0)

    return {
        'loss': total_loss / total_samples,
        'loss_dice': total_dice / total_samples,
    }


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print("Loading training dataset...")
    train_dataset = SolarFilamentFastDataset(
        data_root=args.data_root,
        tile_size=args.tile_size,
        stride=args.stride,
        fg_ratio=args.fg_ratio,
        bnd_ratio=args.bnd_ratio,
        augment=True,  # YOLO-style augmentation
        is_train=True,
        in_channels=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Training dataset size: {len(train_dataset)} tiles")

    # Validation dataset (no augmentation)
    print("Loading validation dataset...")
    val_dataset = SolarFilamentFastDataset(
        data_root=args.data_root,
        tile_size=args.tile_size,
        stride=args.tile_size // 2,  # Overlapping tiles for validation
        fg_ratio=args.fg_ratio,
        bnd_ratio=args.bnd_ratio,
        augment=False,  # No augmentation for validation
        is_train=False,
        in_channels=1,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Validation dataset size: {len(val_dataset)} tiles")

    # Model
    print("Creating segmentation model...")
    model = SolarFilamentSegmentation(
        backbone_name=args.backbone,
        in_channels=1,
        decoder_channels=128,
        pretrained=True,  # ImageNet pretrained
        simmim_checkpoint=args.simmim_checkpoint,  # SimMIM pre-trained
    ).to(device)

    # Loss function
    criterion = CompoundLoss().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # AMP scaler
    scaler = GradScaler('cuda') if args.use_amp else None

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.use_amp,
        )

        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Dice: {train_metrics['loss_dice']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Dice: {val_metrics['loss_dice']:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint_path = save_dir / f"seg_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_dice': val_metrics['loss_dice'],
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if val_metrics['loss_dice'] < best_val_dice:
                best_val_dice = val_metrics['loss_dice']
                best_path = save_dir / "seg_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['loss_dice'],
                }, best_path)
                print(f"Saved best model: {best_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
