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
from pathlib import Path
from tqdm import tqdm


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
    train_parser.add_argument('--tile_size', type=int, default=64,
                             help='Tile size for training patches (reduce for Kaggle memory constraints)')
    train_parser.add_argument('--overlap', type=float, default=0.25,
                             help='Overlap fraction for tiling')
    train_parser.add_argument('--batch_size', type=int, default=1,
                             help='Batch size for training (reduce for Kaggle memory constraints)')
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
        from torch.cuda.amp import GradScaler, autocast
        
        from model import MicroFilNet
        from losses import MicroFilNetLoss
        from dataset import SolarFilamentDataset, create_dataloaders
        from utils import ModelEMA, save_checkpoint, load_checkpoint
        
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
        
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = MicroFilNet().to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        criterion = MicroFilNetLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        ema = ModelEMA(model, decay=0.9999, device=device) if args.use_ema else None
        
        # Import autocast for AMP
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
        train_loader, _ = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            tile_size=args.tile_size,
            num_workers=0,  # Use 0 workers to avoid memory issues on Kaggle
            use_cache=args.use_cache
        )
        
        print(f"Training batches: {len(train_loader)}")
        
        # Training loop
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 50)
            
            model.train()
            total_loss = 0.0
            
            # Add progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
            
            for batch_idx, batch in enumerate(pbar):
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
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            pbar.close()
            print(f"Average loss: {avg_loss:.4f}")
            
            scheduler.step()
            
            # Save checkpoints
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            
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


if __name__ == "__main__":
    main()