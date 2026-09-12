"""
Solar Filament Micro-Segmentation Framework.

Clean, Production-Ready Exports:
  - Model: SolarFilamentNet
  - Dataset: SolarFilamentFastDataset
  - Losses: CompoundLoss, SoftclDiceLoss, SoftDiceLoss
  - Inference: FastPatchInferer
  - Trainer: SolarTrainer
  - Utils: binary_mask_to_rle, rle_to_binary_mask, PanopticQualityMetric
"""

from __future__ import annotations

from src.dataset import SolarFilamentFastDataset
from src.inference import FastPatchInferer
from src.losses import CompoundLoss, SoftclDiceLoss, SoftDiceLoss
from src.model import SolarFilamentNet
from src.trainer import SolarTrainer
from src.utils import (
    ModelEMA,
    PanopticQualityMetric,
    binary_mask_to_rle,
    compute_mean_dice,
    rle_to_binary_mask,
)

__version__ = "2.0.0"

__all__ = [
    "SolarFilamentNet",
    "SolarFilamentFastDataset",
    "CompoundLoss",
    "SoftclDiceLoss",
    "SoftDiceLoss",
    "FastPatchInferer",
    "SolarTrainer",
    "ModelEMA",
    "PanopticQualityMetric",
    "binary_mask_to_rle",
    "compute_mean_dice",
    "rle_to_binary_mask",
]
