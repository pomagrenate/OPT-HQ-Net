"""
Filament-HQ Multi-Head Dense Architecture.

Dense-first model operating on 1024x1024 tiles with four output heads:
  1. Semantic Head  : (B, 1, 1024, 1024) — Filament presence probability.
  2. Boundary Head  : (B, 1, 1024, 1024) — Filament edge/boundary probability map.
  3. Skeleton Head  : (B, 1, 1024, 1024) — Centerline topological spine map.
  4. Instance Head  : (B, 16, 1024, 1024) — Pixel-wise contrastive embeddings (D=16).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filament_hq.models.backbone import BackboneWithFPN4Ch


class DenseHead(nn.Module):
    """
    High-resolution dense prediction head upsampling features 4x back to input size.
    """

    def __init__(self, in_channels: int, out_channels: int, is_embedding: bool = False) -> None:
        super().__init__()
        self.is_embedding = is_embedding
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
        out = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        if self.is_embedding:
            # L2 normalize pixel embeddings across D dimension
            out = F.normalize(out, p=2, dim=1)
        return out


class FilamentHQModel(nn.Module):
    """
    Filament-HQ Dense Multi-Head Model.

    Parameters
    ----------
    backbone_name : str
        Name of backbone (default 'convnext_tiny').
    in_channels : int
        Number of physical input channels (default 4).
    embed_dim : int
        Dimension of pixel-wise instance embedding vectors (default 16).
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        in_channels: int = 4,
        embed_dim: int = 16,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim

        # 1. 4-Channel Backbone + FPN
        self.backbone = BackboneWithFPN4Ch(
            model_name=backbone_name,
            in_channels=in_channels,
            out_channels=128,
            pretrained=pretrained,
        )

        # 2. Dense Task Heads
        self.semantic_head = DenseHead(in_channels=128, out_channels=1, is_embedding=False)
        self.boundary_head = DenseHead(in_channels=128, out_channels=1, is_embedding=False)
        self.skeleton_head = DenseHead(in_channels=128, out_channels=1, is_embedding=False)
        self.instance_head = DenseHead(in_channels=128, out_channels=embed_dim, is_embedding=True)
        self.affinity_head = DenseHead(in_channels=128, out_channels=2, is_embedding=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for 1024x1024 input batch.
        """
        target_size = (x.shape[-2], x.shape[-1])

        # Extract fused P2 feature map (B, 128, H/4, W/4)
        p2_fused = self.backbone(x)

        # Decode dense outputs upsampled to input resolution
        sem_logits = self.semantic_head(p2_fused, target_size)
        bnd_logits = self.boundary_head(p2_fused, target_size)
        skl_logits = self.skeleton_head(p2_fused, target_size)
        inst_embeds = self.instance_head(p2_fused, target_size)
        aff_logits = self.affinity_head(p2_fused, target_size)

        return {
            "semantic": sem_logits,
            "boundary": bnd_logits,
            "skeleton": skl_logits,
            "instance": inst_embeds,
            "embedding": inst_embeds,
            "affinity": aff_logits,
        }
