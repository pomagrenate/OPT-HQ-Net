"""
Filament-HQ Model Factory and Version Router.

Exports:
  - FilamentHQModelV1: Original 3-Head Baseline (Semantic, Boundary, Skeleton)
  - FilamentHQModelV2: Multi-Task 5-Head Model (Semantic, Boundary, Skeleton, 16D Embedding, 2D Affinity Graph)
  - FilamentHQModel: Model factory instantiating V1 or V2 via `version` parameter.
"""

from __future__ import annotations

import torch.nn as nn

from filament_hq.models.v1 import FilamentHQModelV1
from filament_hq.models.v2 import FilamentHQModelV2


def FilamentHQModel(
    backbone: str = "convnext_tiny",
    in_channels: int = 4,
    embed_dim: int = 16,
    pretrained: bool = True,
    version: str = "v2",
) -> nn.Module:
    """
    Factory function returning FilamentHQ Model V1 or V2.

    Parameters
    ----------
    backbone : str
        Backbone architecture name (default 'convnext_tiny').
    in_channels : int
        Input physical channels (default 4).
    embed_dim : int
        Pixel embedding dimension (default 16).
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    version : str
        'v1' for baseline 3-head, 'v2' for multi-task 5-head.

    Returns
    -------
    model : nn.Module
    """
    if str(version).lower() == "v1":
        return FilamentHQModelV1(
            backbone_name=backbone,
            in_channels=in_channels,
            pretrained=pretrained,
        )
    return FilamentHQModelV2(
        backbone_name=backbone,
        in_channels=in_channels,
        embed_dim=embed_dim,
        pretrained=pretrained,
    )


__all__ = ["FilamentHQModel", "FilamentHQModelV1", "FilamentHQModelV2"]
