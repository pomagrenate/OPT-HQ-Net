"""
Training Pipeline Profiling Script

This script profiles the training pipeline to identify bottlenecks.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import SolarFilamentFastDataset, SolarFilamentNet, CompoundLoss

def profile_dataloader(data_root, tile_size=512, stride=384, batch_size=4, num_workers=2):
    """Profile DataLoader performance."""
    print("=" * 60)
    print("PROFILING DATALOADER")
    print("=" * 60)

    dataset = SolarFilamentFastDataset(
        data_root=data_root,
        tile_size=tile_size,
        stride=stride,
        fg_ratio=0.70,
        bnd_ratio=0.20,
        augment=True,
        is_train=True,
    )

    print(f"Dataset size: {len(dataset)} tiles")
    print(f"Tile size: {tile_size}x{tile_size}, Stride: {stride}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    print(f"Total batches per epoch: {len(dataloader)}")

    # Profile data loading time
    times = []
    for i, batch in enumerate(tqdm(dataloader, desc="Profiling DataLoader", total=100)):
        start = time.time()
        # Just measure the time to get the batch
        _ = batch["image"].shape
        end = time.time()
        times.append(end - start)
        if i >= 99:
            break

    avg_time = sum(times) / len(times)
    print(f"\nAverage batch loading time: {avg_time*1000:.2f}ms")
    print(f"Estimated epoch time (data loading only): {avg_time * len(dataloader) / 60:.2f} minutes")

def profile_loss_computation(batch_size=4, tile_size=512):
    """Profile loss computation performance."""
    print("\n" + "=" * 60)
    print("PROFILING LOSS COMPUTATION")
    print("=" * 60)

    # Create dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_images = torch.randn(batch_size, 3, tile_size, tile_size).to(device)
    dummy_masks = torch.randint(0, 2, (batch_size, 1, tile_size, tile_size)).float().to(device)

    model = SolarFilamentNet(backbone_name="resnet34", in_channels=3, pretrained=False).to(device)
    model.eval()

    loss_fn = CompoundLoss().to(device)

    # Profile forward pass
    with torch.no_grad():
        start = time.time()
        logits = model(dummy_images)
        forward_time = time.time() - start

    # Profile loss computation
    with torch.no_grad():
        start = time.time()
        loss_dict = loss_fn(logits, dummy_masks)
        loss_time = time.time() - start

    print(f"Forward pass time: {forward_time*1000:.2f}ms")
    profile_loss_breakdown(loss_fn, dummy_images, dummy_masks, device)

def profile_loss_breakdown(loss_fn, images, masks, device):
    """Profile individual loss components."""
    model = SolarFilamentNet(backbone_name="resnet34", in_channels=3, pretrained=False).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(images)

    # Profile each loss component
    mask_logits = torch.clamp(logits[:, 0:1], min=-10.0, max=10.0)
    skel_logits = torch.clamp(logits[:, 1:2], min=-10.0, max=10.0)
    mask_probs = torch.sigmoid(mask_logits)

    # BCE
    start = time.time()
    loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(mask_logits, masks)
    bce_time = time.time() - start

    # Dice
    start = time.time()
    loss_dice = loss_fn.dice_loss(mask_probs, masks)
    dice_time = time.time() - start

    # clDice (this is the expensive one)
    start = time.time()
    loss_cldice = loss_fn.cldice_loss(mask_probs, masks)
    cldice_time = time.time() - start

    # Skeleton
    start = time.time()
    target_skel = loss_fn.cldice_loss.soft_skeletonize(masks, iters=5)
    skel_time = time.time() - start

    print(f"BCE loss time: {bce_time*1000:.2f}ms")
    print(f"Dice loss time: {dice_time*1000:.2f}ms")
    print(f"clDice loss time: {cldice_time*1000:.2f}ms")
    print(f"Skeletonization time: {skel_time*1000:.2f}ms")
    print(f"\nTotal loss time: {(bce_time + dice_time + cldice_time + skel_time)*1000:.2f}ms")

def profile_tile_stride(data_root, tile_size=512):
    """Profile tile stride impact."""
    print("\n" + "=" * 60)
    print("PROFILING TILE STRIDE IMPACT")
    print("=" * 60)

    strides = [384, 448, 512]
    for stride in strides:
        dataset = SolarFilamentFastDataset(
            data_root=data_root,
            tile_size=tile_size,
            stride=stride,
            fg_ratio=0.70,
            bnd_ratio=0.20,
            augment=True,
            is_train=True,
        )
        print(f"Stride {stride}: {len(dataset)} tiles ({len(dataset) * 100 / 3606:.1f}% of current)")

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Check if data root is provided
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        # Use the MAGFiLO dataset
        data_root = "E:/GithubProjects/solar_filament_seg/MAGFiLO_1.0_Kaggle_2026/train"

    if not Path(data_root).exists():
        print(f"Data root not found: {data_root}")
        print("Using dummy profiling instead...")

        # Run loss computation profiling with dummy data
        profile_loss_computation(batch_size=4, tile_size=512)
    else:
        print(f"Profiling with data from: {data_root}")
        profile_dataloader(data_root, tile_size=512, stride=384, batch_size=4, num_workers=2)
        profile_loss_computation(batch_size=4, tile_size=512)
        profile_tile_stride(data_root, tile_size=512)
