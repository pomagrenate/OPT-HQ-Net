"""
Main interface script for solar filament segmentation.

This script provides an easy-to-use interface for:
1. Choosing training data path
2. Running training
3. Running inference

Usage:
    python main.py
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Solar Filament Segmentation - Main Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --data_root /path/to/MAGFiLO_1.0_Kaggle_2026/train
  
  # Run inference
  python main.py predict --weights checkpoints/best_model.pt --data_root /path/to/test
  
  # Train with custom parameters
  python main.py train --data_root /path/to/train --epochs 100 --batch_size 8 --use_amp
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_root', type=str, required=True,
                             help='Path to training data directory')
    train_parser.add_argument('--use_cache', action='store_true',
                             help='Use cached .npy files if available')
    train_parser.add_argument('--tile_size', type=int, default=256,
                             help='Tile size for training patches (larger = better GPU utilization)')
    train_parser.add_argument('--overlap', type=float, default=0.25,
                             help='Overlap fraction for tiling')
    train_parser.add_argument('--batch_size', type=int, default=4,
                             help='Batch size for training (increase for better GPU utilization)')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                             help='Learning rate')
    train_parser.add_argument('--use_amp', action='store_true',
                             help='Use automatic mixed precision training')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                             help='Directory to save checkpoints')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Resume from checkpoint (path or "last")')
    train_parser.add_argument('--use_ema', action='store_true',
                             help='Use exponential moving average')
    train_parser.add_argument('--device', type=str, default='cuda',
                             help='Device to use (cuda/cpu)')
    train_parser.add_argument('--num_gpus', type=int, default=2,
                             help='Number of GPUs to use for DDP training')
    
    # Inference command
    predict_parser = subparsers.add_parser('predict', help='Run inference')
    predict_parser.add_argument('--weights', type=str, required=True,
                                help='Path to model weights checkpoint')
    predict_parser.add_argument('--data_root', type=str, required=True,
                                help='Path to test data directory')
    predict_parser.add_argument('--use_cache', action='store_true',
                                help='Use cached .npy files if available')
    predict_parser.add_argument('--tile_size', type=int, default=256,
                                help='Tile size for inference')
    predict_parser.add_argument('--overlap', type=float, default=0.25,
                                help='Overlap fraction for tiling')
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                                help='Probability threshold')
    predict_parser.add_argument('--min_area', type=int, default=30,
                                help='Minimum component area')
    predict_parser.add_argument('--output', type=str, default='submission.csv',
                                help='Output CSV file path')
    predict_parser.add_argument('--device', type=str, default='cuda',
                                help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        print("=" * 60)
        print("SOLAR FILAMENT SEGMENTATION - TRAINING")
        print("=" * 60)
        print(f"Data path: {args.data_root}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"AMP: {args.use_amp}")
        print(f"Device: {args.device}")
        print("=" * 60)
        
        # Import training modules
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        
        # Import autocast for AMP
        try:
            from torch.amp import autocast, GradScaler
            use_new_amp = True
        except ImportError:
            from torch.cuda.amp import autocast, GradScaler
            use_new_amp = False
        
        # Setup distributed training if multiple GPUs
        use_ddp = args.num_gpus > 1 and torch.cuda.device_count() >= args.num_gpus
        local_rank = 0
        
        if use_ddp:
            # Check if running with torchrun (proper DDP)
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                print(f"Using DDP with {args.num_gpus} GPUs (torchrun)")
                torch.distributed.init_process_group(backend='nccl')
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                torch.cuda.set_device(local_rank)
                device = torch.device(f'cuda:{local_rank}')
            else:
                # Fallback: if not using torchrun, use single GPU
                print(f"Warning: --num_gpus {args.num_gpus} specified but not running with torchrun")
                print(f"Falling back to single GPU training")
                use_ddp = False
                if args.device == 'cuda' and not torch.cuda.is_available():
                    print("CUDA not available, falling back to CPU")
                    device = torch.device('cpu')
                else:
                    device = torch.device(args.device)
        else:
            if args.device == 'cuda' and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = torch.device('cpu')
            else:
                device = torch.device(args.device)
        
        # Debug: Print actual data path structure
        if not use_ddp or local_rank == 0:
            print(f"Data root path: {args.data_root}")
            print(f"Checking for train directories:")
            for path in [
                Path(args.data_root) / "train" / "train_images",
                Path(args.data_root) / "train_images",
                Path(args.data_root) / "train"
            ]:
                print(f"  {path}: exists={path.exists()}")
        
        from model import MicroFilNet
        from losses import MicroFilNetLoss
        from dataset import SolarFilamentDataset, create_dataloaders, collate_fn
        from utils import ModelEMA, save_checkpoint, load_checkpoint
        
        print(f"Using device: {device}")
        
        # Force CUDA if available and requested
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            if not use_ddp:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            else:
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = MicroFilNet().to(device)
        
        # Wrap with DDP if using multiple GPUs
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = MicroFilNetLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        ema = ModelEMA(model, decay=0.9999, device=device) if args.use_ema else None
        
        # Create gradient scaler for AMP
        try:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler() if args.use_amp else None
            use_new_amp = True
        except ImportError:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler() if args.use_amp else None
            use_new_amp = False
        
        # Load checkpoint if resuming
        start_epoch = 0
        best_loss = float('inf')
        
        if args.resume:
            if args.resume == 'last':
                checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                else:
                    checkpoint_path = checkpoint_dir / 'last.pt'
            else:
                checkpoint_path = Path(args.resume)
            
            if checkpoint_path.exists():
                info = load_checkpoint(checkpoint_path, model, optimizer, ema, scheduler, device)
                start_epoch = info['epoch'] + 1
                best_loss = info['loss']
                print(f"Resumed from epoch {start_epoch}")
        
        # Create dataloaders
        print(f"Loading data from: {args.data_root}")
        
        if use_ddp:
            from torch.utils.data.distributed import DistributedSampler
            full_dataset = SolarFilamentDataset(
                data_root=args.data_root,
                split='train',
                tile_size=args.tile_size,
                overlap=args.overlap,
                use_cache=args.use_cache
            )
            
            # Split dataset for train/val
            dataset_size = len(full_dataset)
            val_size = int(dataset_size * 0.1)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.num_gpus, rank=local_rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=args.num_gpus, rank=local_rank, shuffle=False)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn
            )
        else:
            train_loader, val_loader = create_dataloaders(
                data_root=args.data_root,
                batch_size=args.batch_size,
                tile_size=args.tile_size,
                num_workers=2,
                use_cache=args.use_cache,
                val_split=0.1
            )
        
        print(f"Training batches: {len(train_loader)}")
        
        # Training loop
        for epoch in range(start_epoch, args.epochs):
            if use_ddp:
                train_sampler.set_epoch(epoch)
            
            if not use_ddp or local_rank == 0:
                print(f"\nEpoch {epoch + 1}/{args.epochs}")
                print("-" * 50)
            
            model.train()
            total_loss = 0.0
            
            # Add progress bar (only on rank 0)
            if not use_ddp or local_rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
                iterator = pbar
            else:
                iterator = train_loader
            
            for batch_idx, batch in enumerate(iterator):
                images = batch['image'].to(device)
                valid_masks = batch['valid_mask'].to(device)
                masks = batch['mask'].to(device)
                
                optimizer.zero_grad()
                
                if args.use_amp:
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
                
                if ema:
                    ema.update()
                
                total_loss += loss.item()
                
                # Update progress bar (only on rank 0)
                if not use_ddp or local_rank == 0:
                    iterator.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            if not use_ddp or local_rank == 0:
                iterator.close()
                print(f"Average train loss: {avg_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                if not use_ddp or local_rank == 0:
                    print("Running validation...")
                
                model.eval()
                val_loss = 0.0
                val_samples = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(device)
                        valid_masks = batch['valid_mask'].to(device)
                        masks = batch['mask'].to(device)
                        
                        logits = model(images)
                        loss, parts = criterion(logits, masks, valid_masks, epoch)
                        
                        val_loss += loss.item() * images.size(0)
                        val_samples += images.size(0)
                
                avg_val_loss = val_loss / val_samples
                if not use_ddp or local_rank == 0:
                    print(f"Average val loss: {avg_val_loss:.4f}")
                
                # Use validation loss for best model saving
                current_loss = avg_val_loss
                
                # Create visualization plots
                if not use_ddp or local_rank == 0:
                    create_validation_plots(model, val_loader, device, epoch, checkpoint_dir)
            else:
                current_loss = avg_loss
            
            scheduler.step()
            
            # Save checkpoints (only on rank 0)
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            
            if not use_ddp or local_rank == 0:
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir / 'last.pt', ema, scheduler)
                if is_best:
                    save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir / 'best_model.pt', ema, scheduler)
        
        print("\nTraining completed!")
        print(f"Best loss: {best_loss:.4f}")
        
    elif args.command == 'predict':
        print("=" * 60)
        print("SOLAR FILAMENT SEGMENTATION - INFERENCE")
        print("=" * 60)
        print(f"Model weights: {args.weights}")
        print(f"Data path: {args.data_root}")
        print(f"Threshold: {args.threshold}")
        print(f"Output: {args.output}")
        print(f"Device: {args.device}")
        print("=" * 60)
        
        # Import inference modules
        import torch
        import cv2
        
        from model import MicroFilNet
        from dataset import SolarFilamentDataset
        from inference import tiled_predict, postprocess_mask
        from utils import binary_mask_to_rle, create_submission_csv, load_checkpoint
        
        # Setup
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
        
        # Load model
        model = MicroFilNet().to(device)
        checkpoint_path = Path(args.weights)
        if checkpoint_path.exists():
            load_checkpoint(checkpoint_path, model, device=device)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Create test dataset
        print(f"Loading test data from: {args.data_root}")
        test_dataset = SolarFilamentDataset(
            data_root=args.data_root,
            split='test',
            tile_size=args.tile_size,
            overlap=args.overlap,
            use_cache=args.use_cache
        )
        
        print(f"Test images: {len(test_dataset)}")
        
        # Run inference
        predictions = {}
        model.eval()
        
        with torch.no_grad():
            # Add progress bar for inference
            pbar = tqdm(range(len(test_dataset)), desc="Inference", leave=False)
            for idx in pbar:
                sample = test_dataset[idx]
                image_id = sample['image_id']
                image = sample['image'].numpy()
                valid_mask = sample['valid_mask'].numpy()
                
                pbar.set_postfix({'image': image_id})
                pbar.update(1)
                
                prob_map = tiled_predict(
                    model, image, valid_mask,
                    tile=args.tile_size, overlap=args.overlap, device=device,
                    batch_size=8
                )
                
                binary_mask = postprocess_mask(
                    prob_map, threshold=args.threshold,
                    close_kernel_px=3, min_area_px=args.min_area
                )
                
                # Extract connected components
                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                components = []
                for lbl in range(1, n_labels):
                    if stats[lbl, cv2.CC_STAT_AREA] >= args.min_area:
                        component = (labels == lbl).astype(np.uint8)
                        components.append(component)
                
                # Convert to RLE
                rle_strings = [binary_mask_to_rle(comp) for comp in components]
                predictions[image_id] = rle_strings
        
        pbar.close()
        
        # Create submission
        create_submission_csv(predictions, args.output)
        total_filaments = sum(len(rles) for rles in predictions.values())
        print(f"\nInference completed!")
        print(f"Total images: {len(predictions)}")
        print(f"Total filaments: {total_filaments}")


def create_validation_plots(model, val_loader, device, epoch, checkpoint_dir):
    """Create visualization plots of original vs segmented images."""
    import torch  # Import torch here to avoid namespace issues
    
    model.eval()
    
    # Get a few samples from validation
    samples = []
    with torch.no_grad():
        for batch in val_loader:
            samples.append(batch)
            if len(samples) >= 2:  # Get 2 batches
                break
    
    # Create plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Epoch {epoch + 1} - Validation Results', fontsize=16)
    
    sample_idx = 0
    for batch in samples:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Get predictions
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Plot first 2 samples from this batch
        for i in range(min(2, images.size(0))):
            if sample_idx >= 4:
                break
            
            # Original image (first channel)
            ax = axes[sample_idx // 2, sample_idx % 2 * 2]
            ax.imshow(images[i, 0].cpu().numpy(), cmap='gray')
            ax.set_title('Original')
            ax.axis('off')
            
            # Prediction
            ax = axes[sample_idx // 2, sample_idx % 2 * 2 + 1]
            ax.imshow(preds[i, 0].cpu().numpy(), cmap='gray')
            ax.set_title('Prediction')
            ax.axis('off')
            
            sample_idx += 1
    
    plt.tight_layout()
    plot_path = checkpoint_dir / f'val_epoch_{epoch + 1}.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved validation plot: {plot_path}")


if __name__ == "__main__":
    main()