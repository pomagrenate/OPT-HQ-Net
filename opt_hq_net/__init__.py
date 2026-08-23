"""
OPT-HQ Net: Oriented-Prompted Topological Network for Solar Filament Segmentation.

Package hierarchy:
    opt_hq_net/
    ├── config.py          — Typed dataclass configs (ModelConfig, TrainingConfig, …)
    ├── data/              — Dataset, preprocessing, augmentation
    ├── models/            — Backbone, Oriented RPN, Rotated RoIAlign, Mask Decoder
    ├── losses/            — SkeletonRecall, Focal, Dice, combined loss
    ├── postprocess/       — Rotated NMS, morphological cleaning, RLE encoder
    ├── metrics/           — Panoptic Quality, Dice, IoU
    └── engine/            — Trainer (AMP), Inference pipeline → submission CSV
"""

from opt_hq_net.config import ModelConfig, TrainingConfig, PostProcessConfig
from opt_hq_net.models.opt_hq_net import OPTHQNet, OPTHQNetBuilder

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "PostProcessConfig",
    "OPTHQNet",
    "OPTHQNetBuilder",
]
