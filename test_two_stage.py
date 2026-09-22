"""
Test script for two-stage framework verification.
"""

import torch
from src.simmim import SimMIMSegFormer
from src.segmentation import SolarFilamentSegmentation


def test_simmim():
    """Test SimMIM model."""
    print("=" * 60)
    print("Testing SimMIM Model")
    print("=" * 60)

    model = SimMIMSegFormer(
        backbone_name="nvidia/mit-b0",
        in_channels=1,
        mask_ratio=0.5,
        pretrained=False,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"Mask shape: {outputs['mask'].shape}")
    print(f"Encoder features shape: {outputs['encoder_features'].shape}")

    assert outputs['reconstructed'].shape == (2, 1, 512, 512), "Unexpected reconstruction shape"
    assert outputs['mask'].shape == (2, 1, 512, 512), "Unexpected mask shape"
    print("✓ SimMIM test passed")
    print()


def test_segmentation():
    """Test segmentation model."""
    print("=" * 60)
    print("Testing Segmentation Model")
    print("=" * 60)

    model = SolarFilamentSegmentation(
        backbone_name="nvidia/mit-b0",
        in_channels=1,
        decoder_channels=128,
        pretrained=False,
        simmim_checkpoint=None,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")

    assert logits.shape == (2, 2, 512, 512), "Unexpected output shape"
    print("✓ Segmentation test passed")
    print()


def test_weight_transfer():
    """Test weight transfer from SimMIM to segmentation."""
    print("=" * 60)
    print("Testing Weight Transfer")
    print("=" * 60)

    # Create and save SimMIM checkpoint
    simmim_model = SimMIMSegFormer(
        backbone_name="nvidia/mit-b0",
        in_channels=1,
        mask_ratio=0.5,
        pretrained=False,
    )

    checkpoint_path = "test_simmim_checkpoint.pt"
    torch.save({
        'model': simmim_model.encoder.segformer.state_dict(),
    }, checkpoint_path)

    print(f"Saved SimMIM checkpoint: {checkpoint_path}")

    # Create segmentation model with SimMIM weights
    seg_model = SolarFilamentSegmentation(
        backbone_name="nvidia/mit-b0",
        in_channels=1,
        decoder_channels=128,
        pretrained=False,
        simmim_checkpoint=checkpoint_path,
    )

    print("Loaded SimMIM weights into segmentation model")

    # Clean up
    import os
    os.remove(checkpoint_path)
    print("✓ Weight transfer test passed")
    print()


if __name__ == "__main__":
    test_simmim()
    test_segmentation()
    test_weight_transfer()

    print("=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
