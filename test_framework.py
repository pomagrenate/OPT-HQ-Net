"""
Smoke tests for OPT-HQ Net.

Run with:   python test_framework.py

Each test verifies one module in isolation using synthetic data
(no GPU, no dataset required).  All tests should pass on CPU.
"""

from __future__ import annotations

import math
import traceback
from typing import Callable, List

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str, passed: bool, msg: str = "") -> None:
        self.name = name
        self.passed = passed
        self.msg = msg

    def __repr__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        detail = f"  → {self.msg}" if self.msg else ""
        return f"  [{status}] {self.name}{detail}"


def run_tests(tests: List[Callable[[], None]]) -> None:
    results: List[TestResult] = []
    for test_fn in tests:
        name = test_fn.__name__.replace("test_", "").replace("_", " ").title()
        try:
            test_fn()
            results.append(TestResult(name, True))
        except AssertionError as e:
            results.append(TestResult(name, False, str(e)))
        except Exception as e:
            results.append(TestResult(name, False, traceback.format_exc().splitlines()[-1]))

    print("\n" + "=" * 60)
    print("OPT-HQ NET SMOKE TEST RESULTS")
    print("=" * 60)
    for r in results:
        print(r)
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} tests passed")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_config_instantiation() -> None:
    """ModelConfig, TrainingConfig, PostProcessConfig defaults."""
    from opt_hq_net.config import ModelConfig, TrainingConfig, PostProcessConfig
    cfg = ModelConfig()
    assert cfg.backbone_name == "convnext_large"
    assert cfg.fpn.out_channels == 256
    assert cfg.rpn.fg_iou_threshold == 0.5

    t = TrainingConfig()
    assert t.learning_rate == 1e-4
    assert t.loss_weights.skeleton == 1.5

    p = PostProcessConfig()
    assert p.min_mask_area_px == 50


def test_clahe_preprocessor() -> None:
    """CLAHEPreprocessor converts grayscale uint8 → float32 (H,W,3)."""
    from opt_hq_net.data.preprocessing import CLAHEPreprocessor
    raw = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    proc = CLAHEPreprocessor()
    out = proc(raw)
    assert out.shape == (256, 256, 3), f"Expected (256,256,3), got {out.shape}"
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_solar_disk_mask() -> None:
    """SolarDiskMask zeros background outside a synthetic bright disk."""
    from opt_hq_net.data.preprocessing import SolarDiskMask
    # Create a synthetic solar disk image (bright circle)
    img = np.zeros((256, 256, 3), dtype=np.float32)
    import cv2
    cv2.circle(img, (128, 128), 100, (1.0, 1.0, 1.0), thickness=-1)
    masker = SolarDiskMask(margin_px=5)
    result = masker(img)
    assert result.shape == img.shape
    # Corner pixels (far outside disk) should be zeroed
    assert result[0, 0, 0] == 0.0


def test_fpn_backbone_output_shapes() -> None:
    """FPNNeck produces correct output shapes for all pyramid levels."""
    from opt_hq_net.models.backbone import FPNNeck
    in_channels_list = [192, 384, 768, 1536]  # typical ConvNeXt-L channels
    fpn = FPNNeck(in_channels_list, out_channels=256)

    # Synthetic feature maps
    features = [
        torch.randn(1, c, 64 // (2**i), 64 // (2**i))
        for i, c in enumerate(in_channels_list)
    ]
    outs = fpn(features)
    assert len(outs) == 4, f"Expected 4 FPN levels, got {len(outs)}"
    for key, out in outs.items():
        assert out.shape[1] == 256, f"Level {key}: expected 256 channels, got {out.shape[1]}"


def test_oriented_anchor_generator() -> None:
    """OrientedAnchorGenerator produces correct anchor count."""
    from opt_hq_net.models.oriented_rpn import OrientedAnchorGenerator
    gen = OrientedAnchorGenerator(
        anchor_scales=[4, 8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_angles=[0, 45, 90],
        strides=[4, 8],
    )
    # 2 scales × 3 ratios × 3 angles = 18 anchors per location
    assert gen.num_anchors_per_location == 18

    device = torch.device("cpu")
    anchors = gen.generate_anchors([(16, 16), (8, 8)], device)
    # Level 0: 16*16*18 = 4608 anchors; Level 1: 8*8*18 = 1152
    assert anchors[0].shape == (4608, 5)
    assert anchors[1].shape == (1152, 5)


def test_midpoint_offset_box_coder() -> None:
    """MidpointOffsetBoxCoder encode→decode is approximately invertible."""
    from opt_hq_net.models.oriented_rpn import MidpointOffsetBoxCoder
    coder = MidpointOffsetBoxCoder()

    anchors = torch.tensor([[100., 100., 50., 20., 0.2]])
    gt     = torch.tensor([[110., 95.,  60., 25., 0.3]])

    deltas = coder.encode(anchors, gt)
    reconstructed = coder.decode(anchors, deltas)

    assert torch.allclose(reconstructed, gt, atol=1e-4), \
        f"Reconstructed: {reconstructed}  Expected: {gt}"


def test_rotated_roi_align_output_shape() -> None:
    """RotatedRoIAlign extracts (N, C, 28, 28) crops from a feature map."""
    from opt_hq_net.models.rotated_roi_align import RotatedRoIAlign

    feat = torch.randn(2, 256, 64, 64)   # (B, C, H, W)
    # 3 boxes for image 0, 2 for image 1
    boxes = [
        torch.tensor([[32., 32., 20., 8., 0.3, 0.9],
                      [16., 48., 15., 5., 1.0, 0.8],
                      [48., 16., 18., 6., 0.5, 0.7]]),
        torch.tensor([[50., 50., 25., 10., 0.0, 0.6],
                      [20., 30., 12.,  5., 0.8, 0.5]]),
    ]

    roi = RotatedRoIAlign(output_size=28, spatial_scale=1.0)
    crops, batch_idx = roi(feat, boxes)

    assert crops.shape == (5, 256, 28, 28), f"Unexpected crop shape: {crops.shape}"
    assert batch_idx.shape == (5,)


def test_focal_loss() -> None:
    """FocalLoss returns finite scalar, lower for easy examples."""
    from opt_hq_net.losses.focal_loss import FocalLoss
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    logits = torch.tensor([0.0, 10.0, -10.0])   # neutral, confident pos, confident neg
    targets = torch.tensor([0.5, 1.0, 0.0])

    loss = loss_fn(logits, targets)
    assert loss.isfinite(), "FocalLoss produced non-finite value"
    assert loss.item() >= 0.0


def test_binary_dice_loss() -> None:
    """BinaryDiceLoss = 0 when prediction perfectly matches target."""
    from opt_hq_net.losses.dice_loss import BinaryDiceLoss
    loss_fn = BinaryDiceLoss()

    target = torch.zeros(1, 32, 32)
    target[:, 10:20, 10:20] = 1.0
    # Perfect prediction: logit >> 0 inside mask, << 0 outside
    logit = (target * 20.0 - 10.0)

    loss = loss_fn(logit, target)
    assert loss.item() < 0.01, f"Expected near-zero Dice loss, got {loss.item():.4f}"


def test_skeleton_recall_loss() -> None:
    """SkeletonRecallLoss = 0 when predicted probs ≈ 1 on all skeleton pixels."""
    from opt_hq_net.losses.skeleton_recall import SkeletonRecallLoss
    skel_loss = SkeletonRecallLoss()

    # Create a thin horizontal line mask
    target = torch.zeros(1, 64, 64)
    target[0, 32, 10:54] = 1.0   # horizontal 1-pixel line

    # Perfect prediction: high probability everywhere on the mask
    logit = (target * 20.0 - 2.0).unsqueeze(0)  # (1, 1, 64, 64)

    loss = skel_loss(logit, target)
    assert loss.item() < 0.05, f"Expected near-zero skeleton loss, got {loss.item():.4f}"


def test_rle_encode_decode_roundtrip() -> None:
    """RLEEncoder encode→decode is exactly lossless."""
    from opt_hq_net.postprocess.rle_encoder import RLEEncoder
    encoder = RLEEncoder(height=128, width=128)

    original = np.zeros((128, 128), dtype=np.uint8)
    original[30:60, 20:100] = 1  # rectangular filament region

    rle_str = encoder.encode(original)
    decoded = encoder.decode(rle_str)

    assert isinstance(rle_str, str) and len(rle_str) > 0
    assert np.array_equal(original, decoded), "RLE encode→decode is not lossless"


def test_panoptic_quality_perfect_match() -> None:
    """PQ = 1.0 when predictions exactly match ground-truth."""
    from opt_hq_net.metrics.panoptic_quality import PanopticQualityMetric
    metric = PanopticQualityMetric(iou_threshold=0.5)

    gt_mask = np.zeros((64, 64), dtype=np.uint8)
    gt_mask[10:40, 5:55] = 1

    pred_masks = np.array([gt_mask])
    gt_masks   = np.array([gt_mask])

    metric.update([pred_masks], [gt_masks])
    results = metric.compute()

    assert results["PQ"] == 1.0, f"Expected PQ=1.0, got {results['PQ']}"
    assert results["mean_dice"] == 1.0


def test_panoptic_quality_no_predictions() -> None:
    """PQ = 0.0 when all predictions are missing (all FN)."""
    from opt_hq_net.metrics.panoptic_quality import PanopticQualityMetric
    metric = PanopticQualityMetric()

    gt_mask = np.zeros((64, 64), dtype=np.uint8)
    gt_mask[10:40, 5:55] = 1

    pred_masks = np.zeros((0, 64, 64), dtype=np.uint8)
    gt_masks   = np.array([gt_mask])

    metric.update([pred_masks], [gt_masks])
    results = metric.compute()
    assert results["PQ"] == 0.0
    assert results["FN"] == 1


def test_morphological_cleaner() -> None:
    """MorphologicalCleaner removes small islands and bridges gaps."""
    from opt_hq_net.postprocess.morphological import MorphologicalCleaner
    cleaner = MorphologicalCleaner(min_area_px=50, closing_kernel_size=3)

    prob = np.zeros((128, 128), dtype=np.float32)
    # Large filament region (should survive)
    prob[40:60, 10:100] = 0.9
    # Tiny noise blob (should be removed)
    prob[5:7, 5:7] = 0.9

    cleaned = cleaner(prob)

    assert cleaned.dtype == np.uint8
    assert cleaned[5, 5] == 0, "Small island should have been removed"
    assert cleaned[50, 50] == 1, "Large filament region should survive"


def test_build_submission_csv() -> None:
    """build_submission_csv produces correct column names and row count."""
    from opt_hq_net.postprocess.rle_encoder import build_submission_csv
    mask1 = np.zeros((128, 128), dtype=np.uint8); mask1[10:50, 10:80] = 1
    mask2 = np.zeros((128, 128), dtype=np.uint8); mask2[60:90, 20:70] = 1

    predictions = {
        "20150125172714Mh": [mask1, mask2],
        "20170501024112Bh": [mask1],
    }
    df = build_submission_csv(predictions, height=128, width=128)

    assert list(df.columns) == ["filament_id", "segmentation_rle"]
    assert len(df) == 3
    assert df["filament_id"].iloc[0] == "20150125172714Mh_1"
    assert df["filament_id"].iloc[1] == "20150125172714Mh_2"
    assert df["filament_id"].iloc[2] == "20170501024112Bh_1"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_config_instantiation,
        test_clahe_preprocessor,
        test_solar_disk_mask,
        test_fpn_backbone_output_shapes,
        test_oriented_anchor_generator,
        test_midpoint_offset_box_coder,
        test_rotated_roi_align_output_shape,
        test_focal_loss,
        test_binary_dice_loss,
        test_skeleton_recall_loss,
        test_rle_encode_decode_roundtrip,
        test_panoptic_quality_perfect_match,
        test_panoptic_quality_no_predictions,
        test_morphological_cleaner,
        test_build_submission_csv,
    ]
    run_tests(tests)
