"""
Unit Verification Script for Filament-HQ Framework v1.0.
"""

import torch
import numpy as np
from filament_hq.data.preprocessor import SolarPhysicalPreprocessor
from filament_hq.data.tiler import ImageTiler, TileStitcher
from filament_hq.models.model import FilamentHQModel
from filament_hq.losses.losses import FilamentCompoundLoss


def test_filament_hq_pipeline():
    print("--- [1/4] Testing SolarPhysicalPreprocessor ---")
    dummy_img = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    preprocessor = SolarPhysicalPreprocessor()
    ch4 = preprocessor(dummy_img)
    assert ch4.shape == (2048, 2048, 4), f"Expected (2048, 2048, 4), got {ch4.shape}"
    print(f"✓ 4-channel preprocessor OK. Range: min={ch4.min():.3f}, max={ch4.max():.3f}")

    print("\n--- [2/4] Testing ImageTiler & TileStitcher ---")
    tiler = ImageTiler(tile_size=1024, stride=768)
    tiles, coords = tiler.extract_tiles(ch4)
    print(f"✓ Extracted {len(tiles)} overlapping 1024x1024 tiles from 2048x2048 image.")
    stitcher = TileStitcher(full_shape=(2048, 2048), tile_size=1024, num_channels=1)
    dummy_pred = np.ones((1024, 1024), dtype=np.float32)
    for tile, coord in zip(tiles, coords):
        stitcher.add_tile(dummy_pred, coord)
    stitched = stitcher.get_stitched_map()
    assert stitched.shape == (2048, 2048), f"Expected (2048, 2048), got {stitched.shape}"
    print(f"✓ TileStitcher Gaussian blending OK. Stitched shape: {stitched.shape}")

    print("\n--- [3/4] Testing FilamentHQModel Forward Pass ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FilamentHQModel(backbone_name="resnet34", in_channels=4, embed_dim=16).to(device)
    dummy_tensor = torch.randn(2, 4, 1024, 1024, device=device)
    with torch.no_grad():
        outputs = model(dummy_tensor)

    assert outputs["semantic"].shape == (2, 1, 1024, 1024)
    assert outputs["boundary"].shape == (2, 1, 1024, 1024)
    assert outputs["skeleton"].shape == (2, 1, 1024, 1024)
    assert outputs["instance"].shape == (2, 16, 1024, 1024)
    print("✓ Model Forward Pass OK across all 4 dense heads!")

    print("\n--- [4/4] Testing FilamentCompoundLoss ---")
    loss_fn = FilamentCompoundLoss(stage=1).to(device)
    dummy_batch = {
        "semantic": torch.ones(2, 1, 1024, 1024, device=device),
        "boundary": torch.zeros(2, 1, 1024, 1024, device=device),
        "skeleton": torch.zeros(2, 1, 1024, 1024, device=device),
        "instances": torch.ones(2, 1, 1024, 1024, device=device),
    }
    loss_dict = loss_fn(outputs, dummy_batch)
    print(f"✓ Compound Loss OK! Total Loss: {loss_dict['total_loss'].item():.4f}")

    print("\n========================================================")
    print("      🎉 ALL FILAMENT-HQ PIPELINE VERIFICATIONS PASSED!")
    print("========================================================\n")


if __name__ == "__main__":
    test_filament_hq_pipeline()
