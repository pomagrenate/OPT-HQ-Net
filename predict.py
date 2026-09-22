"""
Inference script for MicroFilNet solar filament segmentation.

Usage:
    python predict.py --weights checkpoints/best_model.pt --data_root /path/to/test/data --output submission.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from model import MicroFilNet
from dataset import SolarFilamentDataset
from inference import tiled_predict, postprocess_mask
from utils import binary_mask_to_rle, create_submission_csv, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with MicroFilNet')
    
    # Model weights
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights checkpoint')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to the test data directory')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use cached .npy files if available')
    
    # Inference parameters
    parser.add_argument('--tile_size', type=int, default=256,
                       help='Tile size for inference')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap fraction for tiling')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for binary mask')
    parser.add_argument('--min_area', type=int, default=30,
                       help='Minimum area for connected components')
    parser.add_argument('--close_kernel', type=int, default=3,
                       help='Kernel size for morphological closing')
    
    # Output
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output CSV file path')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for tile inference')
    
    return parser.parse_args()


def extract_connected_components(mask: np.ndarray, min_area: int = 30) -> List[np.ndarray]:
    """
    Extract individual connected components from binary mask.
    
    Args:
        mask: Binary mask (H, W)
        min_area: Minimum area threshold for components
        
    Returns:
        List of binary masks, one for each component
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    components = []
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            component = (labels == lbl).astype(np.uint8)
            components.append(component)
    
    return components


def run_inference(model: nn.Module, dataset: SolarFilamentDataset, 
                  device: str, tile_size: int = 256, overlap: float = 0.25,
                  threshold: float = 0.5, min_area: int = 30,
                  close_kernel: int = 3, batch_size: int = 8) -> dict:
    """
    Run inference on dataset and generate predictions.
    
    Args:
        model: Trained model
        dataset: Test dataset
        device: Device to run inference on
        tile_size: Tile size for sliding window
        overlap: Overlap fraction
        threshold: Probability threshold
        min_area: Minimum component area
        close_kernel: Morphological closing kernel size
        batch_size: Batch size for tile inference
        
    Returns:
        Dictionary mapping image_id to list of RLE strings
    """
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_id = sample['image_id']
            image = sample['image'].numpy()  # (2, H, W)
            valid_mask = sample['valid_mask'].numpy()  # (1, H, W)
            
            print(f"Processing {image_id} ({idx + 1}/{len(dataset)})")
            
            # Run tiled prediction
            prob_map = tiled_predict(
                model, image, valid_mask,
                tile=tile_size, overlap=overlap, device=device,
                batch_size=batch_size
            )
            
            # Postprocess
            binary_mask = postprocess_mask(
                prob_map, threshold=threshold,
                close_kernel_px=close_kernel, min_area_px=min_area
            )
            
            binary_mask = result['mask']
            
            # Extract connected components (individual filaments)
            components = extract_connected_components(binary_mask, min_area=min_area)
            
            # Convert each component to RLE
            rle_strings = []
            for component in components:
                rle = binary_mask_to_rle(component)
                rle_strings.append(rle)
            
            predictions[image_id] = rle_strings
            
            print(f"  Found {len(components)} filaments")
    
    return predictions


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = MicroFilNet().to(device)
    
    # Load checkpoint
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
    print("\nRunning inference...")
    predictions = run_inference(
        model, test_dataset, device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
        min_area=args.min_area,
        close_kernel=args.close_kernel,
        batch_size=args.batch_size
    )
    
    # Create submission CSV
    print(f"\nCreating submission CSV: {args.output}")
    create_submission_csv(predictions, args.output)
    
    # Print summary
    total_filaments = sum(len(rles) for rles in predictions.values())
    print(f"\nInference completed!")
    print(f"Total images processed: {len(predictions)}")
    print(f"Total filaments detected: {total_filaments}")
    print(f"Average filaments per image: {total_filaments / len(predictions):.2f}")


if __name__ == "__main__":
    main()