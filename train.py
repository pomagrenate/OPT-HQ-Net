"""
Training script for MicroFilNet solar filament segmentation.

Usage:
    python train.py --data_root /path/to/MAGFiLO_1.0_Kaggle_2026/train --epochs 50 --batch_size 4
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MicroFilNet
from losses import MicroFilNetLoss
from dataset import SolarFilamentDataset, create_dataloaders
from utils import ModelEMA, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train MicroFilNet for solar filament segmentation')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to the training data directory')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use cached .npy files if available')
    
    # Model arguments
    parser.add_argument('--tile_size', type=int, default=64,
                       help='Tile size for training patches (smaller = faster)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap fraction for tiling')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training (smaller = faster)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Training options
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision training')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (or "last" for latest)')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # EMA
    parser.add_argument('--use_ema', action='store_true',
                       help='Use exponential moving average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA decay rate')
    
    return parser.parse_args()


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: str, epoch: int,
                use_amp: bool = False, scaler: Optional[GradScaler] = None,
                ema: Optional[ModelEMA] = None, use_new_amp: bool = False) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_components = {
        'bce': 0.0,
        'dice': 0.0,
        'cldice': 0.0,
        'boundary': 0.0
    }
    
    num_batches = len(dataloader)
    
    # Add progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        valid_masks = batch['valid_mask'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            if use_new_amp:
                with autocast(device_type='cuda'):
                    logits = model(images)
                    loss, parts = criterion(logits, masks, valid_masks, epoch)
            else:
                with autocast():
                    logits = model(images)
                    loss, parts = criterion(logits, masks, valid_masks, epoch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, parts = criterion(logits, masks, valid_masks, epoch)
            loss.backward()
            optimizer.step()
        
        # Update EMA if enabled
        if ema is not None:
            ema.update()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += parts[key].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bce': f'{parts["bce"]:.4f}',
            'dice': f'{parts["dice"]:.4f}'
        })
    
    # Average losses
    pbar.close()
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return {
        'total_loss': avg_loss,
        **loss_components
    }


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: str, epoch: int) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    loss_components = {
        'bce': 0.0,
        'dice': 0.0,
        'cldice': 0.0,
        'boundary': 0.0
    }
    
    num_batches = len(dataloader)
    
    with torch.no_grad():
        # Add progress bar for validation
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            valid_masks = batch['valid_mask'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            loss, parts = criterion(logits, masks, valid_masks, epoch)
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += parts[key].item()
            
            # Update progress bar
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    # Average losses
    pbar.close()
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return {
        'total_loss': avg_loss,
        **loss_components
    }


def main():
    args = parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Force CUDA if available and requested
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create model
    model = MicroFilNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = MicroFilNetLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create EMA if enabled
    ema = None
    if args.use_ema:
        ema = ModelEMA(model, decay=args.ema_decay, device=device)
    
    # Create gradient scaler for AMP
    use_new_amp = False
    scaler = None
    if args.use_amp:
        try:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler()
            use_new_amp = True
        except ImportError:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if args.resume == 'last':
            # Find the most recent checkpoint
            checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=os.path.getctime)
            else:
                checkpoint_path = checkpoint_dir / 'last.pt'
        else:
            checkpoint_path = Path(args.resume)
        
        if checkpoint_path.exists():
            info = load_checkpoint(
                checkpoint_path, model, optimizer, ema, scheduler, device
            )
            start_epoch = info['epoch'] + 1
            best_loss = info['loss']
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Create dataloaders
    print(f"Loading data from: {args.data_root}")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        tile_size=args.tile_size,
        num_workers=args.num_workers,
        use_cache=args.use_cache
    )
    
    print(f"Training batches: {len(train_loader)}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_amp=args.use_amp, scaler=scaler, ema=ema, use_new_amp=use_new_amp
        )
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Validate if validation loader is available
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device, epoch)
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            current_loss = val_metrics['total_loss']
        else:
            current_loss = train_metrics['total_loss']
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoints
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
        
        # Save last checkpoint
        save_checkpoint(
            model, optimizer, epoch, current_loss,
            checkpoint_dir / 'last.pt', ema, scheduler
        )
        
        # Save best model
        if is_best:
            save_checkpoint(
                model, optimizer, epoch, current_loss,
                checkpoint_dir / 'best_model.pt', ema, scheduler
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, current_loss,
                checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt', ema, scheduler
            )
    
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()