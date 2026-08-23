"""
Configuration dataclasses for OPT-HQ Net.

All hyper-parameters are centralised here so that every module is
instantiated from a single source of truth.  Configs are plain
dataclasses — JSON-serialisable and IDE-friendly.

Usage
-----
>>> from opt_hq_net.config import ModelConfig, TrainingConfig, PostProcessConfig
>>> cfg = ModelConfig()          # sensible defaults
>>> cfg.backbone_name = "convnext_large"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Model architecture config
# ---------------------------------------------------------------------------

@dataclass
class FPNConfig:
    """Feature Pyramid Network settings."""
    out_channels: int = 256
    """Channel width for every FPN level (P2 … P5)."""
    num_outs: int = 4
    """Number of output pyramid levels (P2=0, P3=1, P4=2, P5=3)."""


@dataclass
class OrientedRPNConfig:
    """Oriented Region Proposal Network settings."""
    anchor_scales: List[int] = field(default_factory=lambda: [8])
    """Anchor scale per FPN level (FPN strides handle multi-scale P2-P5)."""
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    """Aspect ratios w/h."""
    anchor_angles: List[float] = field(default_factory=lambda: [0, 45, 90, 135])
    """Anchor angles in degrees (12 total anchors per location instead of 150)."""
    pre_nms_top_n_train: int = 1000
    pre_nms_top_n_test: int = 500
    post_nms_top_n_train: int = 300
    post_nms_top_n_test: int = 200
    nms_iou_threshold: float = 0.7
    """Threshold for axis-aligned NMS during proposal filtering."""
    fg_iou_threshold: float = 0.5
    bg_iou_threshold: float = 0.3
    batch_size_per_image: int = 256
    positive_fraction: float = 0.5


@dataclass
class RotatedRoIAlignConfig:
    """Rotated RoI feature extraction settings."""
    output_size: int = 28
    """Side length of the square output feature map (28 × 28)."""
    sampling_ratio: int = 2
    """Bilinear sampling points per bin."""


@dataclass
class MaskDecoderConfig:
    """HQ Mask Decoder settings."""
    num_mask_tokens: int = 1
    transformer_dim: int = 256
    transformer_depth: int = 2
    transformer_heads: int = 8
    hq_token_channels: int = 64
    """Inter-scale feature fusion channels."""
    p2_channels: int = 256
    """Channels of P2 feature map fused via the HQ token."""
    num_layers: int = 2
    """Number of transformer decoder layers."""
    num_heads: int = 8
    """Number of attention heads."""
    mlp_dim: int = 2048
    """FFN hidden dimension in transformer layer."""


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    # Backbone
    backbone_name: str = "convnext_small"
    """timm model name. Recommended: 'convnext_small', 'resnet50d', 'resnet34d'."""
    backbone_pretrained: bool = True
    image_size: Tuple[int, int] = (2048, 2048)

    # Sub-module configs
    fpn: FPNConfig = field(default_factory=FPNConfig)
    rpn: OrientedRPNConfig = field(default_factory=OrientedRPNConfig)
    roi_align: RotatedRoIAlignConfig = field(default_factory=RotatedRoIAlignConfig)
    decoder: MaskDecoderConfig = field(default_factory=MaskDecoderConfig)

    # Number of foreground classes (always 1 for filaments)
    num_classes: int = 1


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class LossWeightConfig:
    """λ weights for the compound multi-task loss."""
    oriented_box: float = 1.0    # λ1
    focal: float = 2.0           # λ2
    dice: float = 2.0            # λ3
    skeleton: float = 1.5        # λ4


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters."""
    num_epochs: int = 50
    batch_size: int = 1
    """Per-GPU batch size (set to 1 for 16GB VRAM GPUs)."""
    gradient_accumulation_steps: int = 2
    """Simulate larger batch size (effective batch = batch_size * gradient_accumulation_steps)."""
    num_workers: int = 2

    # Optimiser
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Scheduler
    scheduler: str = "cosine"
    """Supported: 'cosine', 'step'."""
    warmup_epochs: int = 5

    # AMP & Multi-GPU
    use_amp: bool = True
    """Automatic Mixed Precision (FP16)."""
    use_multi_gpu: bool = False
    """Enable nn.DataParallel across multiple GPUs. Default False for maximum speed with list targets."""

    # Augmentation
    aug_min_scale: float = 0.8
    aug_max_scale: float = 1.2
    aug_rotation_degrees: float = 360.0
    aug_crop_sizes: List[int] = field(default_factory=lambda: [1024, 2048])

    # Loss weights
    loss_weights: LossWeightConfig = field(default_factory=LossWeightConfig)

    # Checkpointing & Validation
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    val_every_n_epochs: int = 1
    """Validation frequency in epochs (default: 1 = validate every epoch)."""

    # Device
    device: str = "cuda"
    """'cuda' or 'cpu'."""


# ---------------------------------------------------------------------------
# Post-processing config
# ---------------------------------------------------------------------------

@dataclass
class PostProcessConfig:
    """Post-processing pipeline settings."""
    rotated_nms_iou_threshold: float = 0.40
    """Polygon IoU threshold for Rotated NMS."""
    min_mask_area_px: int = 50
    """Connected components smaller than this are discarded as noise."""
    closing_kernel_size: int = 3
    """Morphological closing kernel for bridging faint spine gaps."""
    score_threshold: float = 0.05
    """Minimum predicted confidence to keep a proposal."""
    image_height: int = 2048
    image_width: int = 2048
