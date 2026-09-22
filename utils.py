"""
Utility functions for RLE encoding/decoding and other helper functions.

Includes:
- Column-major RLE encoding/decoding (Kaggle format)
- Panoptic Quality metric
- Model EMA (Exponential Moving Average)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import pandas as pd


def binary_mask_to_rle(mask: np.ndarray) -> str:
    """
    Convert binary mask to column-major run-length encoding (RLE) string.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        
    Returns:
        RLE string in column-major format (Kaggle competition format)
    """
    assert mask.ndim == 2, "Mask must be 2D"
    assert mask.dtype in [np.uint8, np.int32, np.int64, np.bool_, bool], "Mask must be binary"
    
    # Convert to boolean if needed
    mask = mask.astype(bool)
    
    # Get dimensions
    h, w = mask.shape
    
    # Flatten in column-major order (Fortran order)
    mask_flat = mask.flatten(order='F')
    
    # Find transitions
    pixels = np.concatenate([[0], mask_flat, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    # Convert to string
    rle_str = ' '.join(str(x) for x in runs)
    
    return rle_str


def rle_to_binary_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """
    Convert column-major RLE string back to binary mask.
    
    Args:
        rle_str: RLE string in column-major format
        height: Height of the output mask
        width: Width of the output mask
        
    Returns:
        Binary mask (H, W) with values 0 or 1
    """
    # Parse RLE string
    runs = list(map(int, rle_str.split()))
    
    # Create mask
    mask = np.zeros(height * width, dtype=np.uint8)
    
    # Fill mask using column-major order
    current = 0
    for i, run in enumerate(runs):
        if i % 2 == 1:  # Fill odd-indexed runs (these are the 1s)
            mask[current:current + run] = 1
        current += run
    
    # Reshape to original dimensions (column-major)
    mask = mask.reshape((width, height), order='F').T
    
    return mask


def compute_panoptic_quality(pred_masks: List[np.ndarray], 
                            gt_masks: List[np.ndarray],
                            iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) metric for segmentation.
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with PQ, SQ, RQ and matched stats
    """
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-8)
    
    # Match predictions to ground truth
    matched_pred = set()
    matched_gt = set()
    iou_sum = 0.0
    
    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0.0
        best_j = -1
        
        for j, gt_mask in enumerate(gt_masks):
            if j in matched_gt:
                continue
            iou = compute_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        
        if best_iou >= iou_threshold:
            matched_pred.add(i)
            matched_gt.add(best_j)
            iou_sum += best_iou
    
    # Compute metrics
    tp = len(matched_pred)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - len(matched_gt)
    
    sq = iou_sum / (tp + 1e-8) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    pq = sq * rq
    
    return {
        'PQ': pq,
        'SQ': sq,
        'RQ': rq,
        'TP': tp,
        'FP': fp,
        'FN': fn
    }


class ModelEMA:
    """
    Model Exponential Moving Average (EMA)
    
    Maintains a moving average of model parameters for more stable inference.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = 'cpu'):
        self.decay = decay
        self.device = device
        
        # Store parameter references
        self.model = model
        self.shadow = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)
    
    def update(self):
        """Update EMA parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to the given model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def state_dict(self):
        """Return state dict for saving."""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, filepath: str,
                   ema_model: Optional[ModelEMA] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save the checkpoint
        ema_model: Optional EMA model to save
        scheduler: Optional learning rate scheduler state
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if ema_model is not None:
        checkpoint['ema_shadow'] = ema_model.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   ema_model: Optional[ModelEMA] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   device: str = 'cpu') -> Dict:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        ema_model: Optional EMA model to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load to
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if ema_model is not None and 'ema_shadow' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_shadow'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    info = {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }
    
    print(f"Checkpoint loaded from {filepath}, epoch {info['epoch']}, loss {info['loss']:.4f}")
    return info


def create_submission_csv(predictions: Dict[str, List[str]], output_path: str):
    """
    Create Kaggle submission CSV from predictions.
    
    Args:
        predictions: Dictionary mapping image_id to list of RLE strings for each filament
        output_path: Path to save the CSV file
    """
    rows = []
    for image_id, rle_list in predictions.items():
        for i, rle_str in enumerate(rle_list):
            filament_id = f"{image_id}_{i+1}"
            rows.append({
                'filament_id': filament_id,
                'segmentation_rle': rle_str
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Submission CSV saved to {output_path} with {len(rows)} rows")


if __name__ == "__main__":
    # Test RLE encoding/decoding
    print("Testing RLE encoding/decoding...")
    
    # Create a simple test mask
    test_mask = np.array([
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint8)
    
    print("Original mask:")
    print(test_mask)
    
    # Encode
    rle_str = binary_mask_to_rle(test_mask)
    print(f"\nRLE string: {rle_str}")
    
    # Decode
    recovered_mask = rle_to_binary_mask(rle_str, test_mask.shape[0], test_mask.shape[1])
    print("\nRecovered mask:")
    print(recovered_mask)
    
    # Verify
    assert np.array_equal(test_mask, recovered_mask), "RLE roundtrip failed!"
    print("✓ RLE encoding/decoding test passed!")
    
    # Test PQ metric
    print("\nTesting Panoptic Quality metric...")
    pred_masks = [test_mask]
    gt_masks = [test_mask]
    pq_metrics = compute_panoptic_quality(pred_masks, gt_masks)
    print(f"PQ metrics: {pq_metrics}")
    print("✓ PQ metric test passed!")
    
    print("\nUtility functions module loaded successfully!")