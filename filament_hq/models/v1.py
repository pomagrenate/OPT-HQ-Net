"""
Filament-HQ Architecture V1 (Baseline 3-Head Dense Model).

Dense-first model operating on 1024x1024 tiles with three output heads:
  1. Semantic Head : (B, 1, 1024, 1024) — Filament presence probability.
  2. Boundary Head : (B, 1, 1024, 1024) — Filament edge/boundary probability map.
  3. Skeleton Head : (B, 1, 1024, 1024) — Centerline topological spine map.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filament_hq.models.backbone import BackboneWithFPN4Ch


class DenseHeadV1(nn.Module):
    """V1 High-resolution dense prediction head."""

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        feat = self.conv(x)
        return F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)


class FilamentHQModelV1(nn.Module):
    """
    Filament-HQ V1 3-Head Architecture.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        in_channels: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name

        # 4-Channel Backbone + FPN
        self.backbone = BackboneWithFPN4Ch(
            model_name=backbone_name,
            in_channels=in_channels,
            out_channels=128,
            pretrained=pretrained,
        )

        # Three V1 Output Heads
        self.semantic_head = DenseHeadV1(in_channels=128, out_channels=1)
        self.boundary_head = DenseHeadV1(in_channels=128, out_channels=1)
        self.skeleton_head = DenseHeadV1(in_channels=128, out_channels=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (x.shape[-2], x.shape[-1])
        p2_fused = self.backbone(x)

        return {
            "semantic": self.semantic_head(p2_fused, target_size),
            "boundary": self.boundary_head(p2_fused, target_size),
            "skeleton": self.skeleton_head(p2_fused, target_size),
        }
