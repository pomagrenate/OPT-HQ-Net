"""
SimMIM Pre-training Script.

Usage:
    python train_simmim.py --data_root <path> --epochs 35 --batch_size 24
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
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
    parser.add_argument("--val_freq", type=int, default=2, help="Validation frequency (epochs)")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio (5% of data)")
    parser.add_argument("--vis_dir", type=str, default="visualizations", help="Visualization directory")
    parser.add_argument("--use_edge_loss", action="store_true", help="Add Sobel edge loss for sharpness")
    parser.add_argument("--edge_loss_weight", type=float, default=0.1, help="Edge loss weight")
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
    use_edge_loss: bool = False,
    edge_loss_weight: float = 0.1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_recon_loss = 0.0
    total_edge_loss = 0.0

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
                recon_loss = F.l1_loss(
                    reconstructed * mask,
                    images * mask,
                    reduction='mean'
                )
                
                # Edge loss for sharpness
                if use_edge_loss:
                    edge_loss = sobel_edge_loss(
                        reconstructed * mask,
                        images * mask
                    )
                    loss = recon_loss + edge_loss_weight * edge_loss
                else:
                    loss = recon_loss
                    edge_loss = torch.tensor(0.0)
        else:
            outputs = model(images, mask=mask)
            reconstructed = outputs['reconstructed']
            mask = outputs['mask']

            recon_loss = F.l1_loss(
                reconstructed * mask,
                images * mask,
                reduction='mean'
            )
            
            if use_edge_loss:
                edge_loss = sobel_edge_loss(
                    reconstructed * mask,
                    images * mask
                )
                loss = recon_loss + edge_loss_weight * edge_loss
            else:
                loss = recon_loss
                edge_loss = torch.tensor(0.0)

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
        total_recon_loss += recon_loss.item() * images.size(0)
        total_edge_loss += edge_loss.item() * images.size(0)
        total_samples += images.size(0)

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{recon_loss.item():.4f}",
            'edge': f"{edge_loss.item():.4f}" if use_edge_loss else "N/A",
        })

    return {
        'loss': total_loss / total_samples,
        'recon_loss': total_recon_loss / total_samples,
        'edge_loss': total_edge_loss / total_samples if use_edge_loss else 0.0,
    }


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel edge loss to encourage sharp predictions.
    
    Parameters
    ----------
    pred : Tensor (B, 1, H, W)
        Predicted reconstruction
    target : Tensor (B, 1, H, W)
        Target image
        
    Returns
    -------
    edge_loss : Tensor
        L1 loss on Sobel edges
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    
    # Compute edges
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
    target_edge_x = F.conv2d(target, sobel_x, padding=1)
    target_edge_y = F.conv2d(target, sobel_y, padding=1)
    
    pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
    target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)
    
    return F.l1_loss(pred_edge, target_edge, reduction='mean')


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate reconstruction on holdout set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)

            # Generate random mask
            mask = model.random_masking(images, model.mask_ratio)

            # Forward pass
            outputs = model(images, mask=mask)
            reconstructed = outputs['reconstructed']
            mask = outputs['mask']

            # L1 loss on masked regions only
            loss = F.l1_loss(
                reconstructed * mask,
                images * mask,
                reduction='mean'
            )

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    model.train()
    return {'loss': total_loss / total_samples}


def visualize_reconstruction(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    vis_dir: Path,
    num_samples: int = 4,
) -> None:
    """
    Visualize reconstruction comparisons with composite inpainting.

    Creates a figure with 3 columns:
    - Column 1: Original image
    - Column 2: Masked input
    - Column 3: Composite reconstruction (original * (1-mask) + pred * mask)
    """
    model.eval()
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch of samples
    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    # Select random samples
    indices = torch.randperm(images.size(0))[:num_samples]
    sample_images = images[indices]

    # Generate masks
    with torch.no_grad():
        masks = model.random_masking(sample_images, model.mask_ratio)
        outputs = model(sample_images, mask=masks)
        reconstructed = outputs['reconstructed']
        mask = outputs['mask']

    # Convert to numpy for plotting
    sample_images_np = sample_images.cpu().numpy()
    masked_images_np = (sample_images * (1 - mask)).cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    mask_np = mask.cpu().numpy()

    # Create composite reconstruction (MAE/SimMIM style)
    # composite = original * (1 - mask) + reconstruction * mask
    composite_np = sample_images_np * (1 - mask_np) + reconstructed_np * mask_np

    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(sample_images_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Masked
        axes[i, 1].imshow(masked_images_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Masked Input')
        axes[i, 1].axis('off')

        # Composite reconstruction
        axes[i, 2].imshow(composite_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Composite Reconstruction')
        axes[i, 2].axis('off')

    plt.suptitle(f'Epoch {epoch} - Reconstruction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    save_path = vis_dir / f"mim_recon_epoch_{epoch:02d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")

    # Display inline for Kaggle/Jupyter
    plt.show()
    plt.close()

    model.train()



def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization directory
    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    print("Loading dataset...")
    full_dataset = SolarFilamentFastDataset(
        data_root=args.data_root,
        tile_size=args.tile_size,
        stride=args.stride,
        fg_ratio=0.0,  # Use all tiles for SimMIM
        bnd_ratio=0.0,
        augment=False,  # No augmentation for SimMIM
        is_train=True,
        in_channels=1,
    )

    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size

    # Create train dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Total dataset size: {total_size} tiles")
    print(f"Training dataset size: {train_size} tiles")
    print(f"Validation dataset size: {val_size} tiles")

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
                start_factor=0.01,  # Must be > 0
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
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Estimated time per epoch: ~{len(train_loader) / 800:.1f} minutes (at 800 it/s)")
    print(f"Estimated total time: ~{len(train_loader) * args.epochs / 800 / 60:.1f} hours")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.use_amp,
            args.use_edge_loss,
            args.edge_loss_weight,
        )

        print(f"Train Loss: {metrics['loss']:.4f} (Recon: {metrics['recon_loss']:.4f}, Edge: {metrics['edge_loss']:.4f})")

        # Validate every val_freq epochs
        if epoch % args.val_freq == 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Val Loss: {val_metrics['loss']:.4f}")

            # Visualize reconstruction
            visualize_reconstruction(
                model,
                val_loader,
                device,
                epoch,
                vis_dir,
                num_samples=4,
            )
            
            # Update best loss based on validation
            current_val_loss = val_metrics['loss']
        else:
            current_val_loss = metrics['loss']

        # Step scheduler after each epoch
        scheduler.step()

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            checkpoint_path = save_dir / f"simmim_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': metrics['loss'],
                'val_loss': val_metrics.get('loss', metrics['loss']),
                'lr': scheduler.get_last_lr()[0],
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if current_val_loss < best_loss:
                best_loss = current_val_loss
                best_path = save_dir / "simmim_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': metrics['loss'],
                    'val_loss': current_val_loss,
                }, best_path)
                print(f"Saved best model: {best_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
