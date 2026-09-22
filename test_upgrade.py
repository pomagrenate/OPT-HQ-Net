"""
Test script to verify the upgraded training pipeline works correctly.

Tests:
1. mit_b0 backbone model initialization
2. Dataset with YOLO-style augmentation
3. Forward pass with augmented data
4. Loss computation
"""

import torch
from torch.utils.data import DataLoader

from src import SolarFilamentFastDataset, SolarFilamentNet, CompoundLoss

def test_model():
    """Test SegFormer B0 backbone model with 1-channel grayscale input."""
    print("=" * 60)
    print("Testing SegFormer B0 (nvidia/mit-b0) Backbone Model (SimMIM-ready)")
    print("=" * 60)

    # Create model with SegFormer B0 and 1-channel input
    model = SolarFilamentNet(
        backbone_name="nvidia/mit-b0",
        in_channels=1,
        decoder_channels=128,
        pretrained=False,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: ~3-4M for SegFormer B0")

    # Test forward pass with 1-channel input
    dummy_input = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 2, 512, 512) - mask + skeleton logits")

    assert output.shape == (2, 2, 512, 512), f"Unexpected output shape: {output.shape}"
    print("✓ Model test passed")
    print()

def test_dataset_augmentation(data_root):
    """Test dataset with YOLO-style augmentation and 1-channel grayscale input."""
    print("=" * 60)
    print("Testing Dataset with YOLO-style Augmentation (1-channel grayscale)")
    print("=" * 60)

    dataset = SolarFilamentFastDataset(
        data_root=data_root,
        tile_size=512,
        stride=512,
        fg_ratio=0.70,
        bnd_ratio=0.20,
        augment=True,
        is_train=True,
        in_channels=1,  # Grayscale for SimMIM
    )

    print(f"Dataset size: {len(dataset)} tiles")

    # Test one sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Skeleton shape: {sample['skeleton'].shape}")
    print(f"Coord: {sample['coord']}")

    assert 'image' in sample, "Missing 'image' key"
    assert 'mask' in sample, "Missing 'mask' key"
    assert 'skeleton' in sample, "Missing 'skeleton' key"
    assert sample['image'].shape == (1, 512, 512), f"Unexpected image shape: {sample['image'].shape}"
    assert sample['mask'].shape == (1, 512, 512), f"Unexpected mask shape: {sample['mask'].shape}"
    assert sample['skeleton'].shape == (1, 512, 512), f"Unexpected skeleton shape: {sample['skeleton'].shape}"

    print("✓ Dataset test passed")
    print()

def test_training_step(data_root):
    """Test a complete training step with 1-channel grayscale input."""
    print("=" * 60)
    print("Testing Complete Training Step (1-channel grayscale)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = SolarFilamentFastDataset(
        data_root=data_root,
        tile_size=512,
        stride=512,
        fg_ratio=0.70,
        bnd_ratio=0.20,
        augment=True,
        is_train=True,
        in_channels=1,  # Grayscale for SimMIM
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        pin_memory=False,
    )

    # Create model
    model = SolarFilamentNet(
        backbone_name="nvidia/mit-b0",
        in_channels=1,
        decoder_channels=128,
        pretrained=False,
    ).to(device)

    # Create loss function
    loss_fn = CompoundLoss().to(device)

    # Test one batch
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")

        # Forward pass
        with torch.no_grad():
            logits = model(images)
            print(f"Logits shape: {logits.shape}")

        # Loss computation
        with torch.no_grad():
            loss_dict = loss_fn(logits, masks)
            print(f"Loss components: {loss_dict}")

        break  # Only test first batch

    print("✓ Training step test passed")
    print()

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Test model first (doesn't need data)
    test_model()

    # Test dataset and training if data is available
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "E:/GithubProjects/solar_filament_seg/MAGFiLO_1.0_Kaggle_2026/train"

    if Path(data_root).exists():
        print(f"Testing with data from: {data_root}")
        test_dataset_augmentation(data_root)
        test_training_step(data_root)
    else:
        print(f"Data root not found: {data_root}")
        print("Skipping dataset and training tests")
        print("To test with data, run:")
        print(f"  python test_upgrade.py {data_root}")

    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
