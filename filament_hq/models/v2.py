"""
Filament-HQ Architecture V2 (Multi-Task 5-Head Instance Model).

Operates on 1024x1024 tiles with five output heads:
  1. Semantic Head  : (B, 1, 1024, 1024) — Filament presence probability.
  2. Boundary Head  : (B, 1, 1024, 1024) — Filament edge/boundary probability map.
  3. Skeleton Head  : (B, 1, 1024, 1024) — Centerline topological spine map.
  4. Embedding Head : (B, 16, 1024, 1024) — L2-normalized 16D pixel cluster embeddings.
  5. Affinity Head  : (B, 2, 1024, 1024) — Horizontal & Vertical pixel connectivity graph.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from filament_hq.models.backbone import BackboneWithFPN4Ch
from filament_hq.models.heads import ConvHead, EmbeddingHead, AffinityHead


class DenseHeadV2(nn.Module):
    """V2 High-resolution dense prediction head."""

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
            out = F.normalize(out, p=2, dim=1)
        return out


class FilamentHQModelV2(nn.Module):
    """
    Filament-HQ V2 5-Head Architecture with Embedding & Affinity Graph.
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

        # 4-Channel Backbone + FPN
        self.backbone = BackboneWithFPN4Ch(
            model_name=backbone_name,
            in_channels=in_channels,
            out_channels=128,
            pretrained=pretrained,
        )

        # Five V2 Output Heads
        self.semantic_head = DenseHeadV2(in_channels=128, out_channels=1, is_embedding=False)
        self.boundary_head = DenseHeadV2(in_channels=128, out_channels=1, is_embedding=False)
        self.skeleton_head = DenseHeadV2(in_channels=128, out_channels=1, is_embedding=False)
        self.instance_head = DenseHeadV2(in_channels=128, out_channels=embed_dim, is_embedding=True)
        self.affinity_head = DenseHeadV2(in_channels=128, out_channels=2, is_embedding=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        target_size = (x.shape[-2], x.shape[-1])
        p2_fused = self.backbone(x)

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
