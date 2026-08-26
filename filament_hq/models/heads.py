"""
Multi-Task Prediction Heads for Filament-HQ Architecture.

Contains:
  1. SemanticHead  -> [B, 1, H, W]
  2. BoundaryHead  -> [B, 1, H, W]
  3. SkeletonHead  -> [B, 1, H, W]
  4. EmbeddingHead -> [B, 16, H, W] (Instance contrastive cluster embeddings)
  5. AffinityHead  -> [B, 2, H, W]  (Horizontal & Vertical pixel instance connectivity)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvHead(nn.Module):
    """Generic Convolutional Prediction Head."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 64) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmbeddingHead(nn.Module):
    """16-D Pixel Embedding Head for Instance Contrastive Clustering."""

    def __init__(self, in_channels: int, embed_dim: int = 16) -> None:
        super().__init__()
        self.head = ConvHead(in_channels, embed_dim, mid_channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class AffinityHead(nn.Module):
    """2-D Neighbor Affinity Head (Ch0: Horizontal, Ch1: Vertical)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = ConvHead(in_channels, out_channels=2, mid_channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
