"""
Ultra-Efficient, Lightweight Neural Architecture for Solar Filament Micro-Segmentation.

Architecture:
  - Backbone: High-throughput timm encoder (ResNet34, ConvNeXt-Nano, or mit_b0)
  - Decoder: Lightweight multi-scale Feature Pyramid Decoder (128 channels)
  - Output: 2-Channel High-Resolution Logits:
      Channel 0: Filament Mask Logits
      Channel 1: Filament Topological Skeleton Logits
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Depthwise-separable or standard 3x3 conv with GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        num_groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SolarFilamentNet(nn.Module):
    """
    Lightweight, High-Precision Micro-Segmentation Network.

    Parameters
    ----------
    backbone_name : str
        Encoder model from timm (default: 'resnet34').
        Options: 'resnet34', 'resnet18', 'convnext_nano', 'mit_b0'.
    in_channels : int
        Number of input image channels (default: 3).
    decoder_channels : int
        Intermediate feature channels in the decoder (default: 128).
    pretrained : bool
        Whether to initialize encoder with ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        in_channels: int = 3,
        decoder_channels: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels = in_channels

        # Instantiate timm feature extractor
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # Strides 4, 8, 16, 32
        )

        enc_channels: List[int] = self.encoder.feature_info.channels()
        c2, c3, c4, c5 = enc_channels

        # Lateral 1x1 projections to decoder_channels
        self.lat_c5 = nn.Conv2d(c5, decoder_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4, decoder_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.lat_c2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)

        # Smooth fusion blocks
        self.smooth_c4 = ConvBlock(decoder_channels, decoder_channels)
        self.smooth_c3 = ConvBlock(decoder_channels, decoder_channels)
        self.smooth_c2 = ConvBlock(decoder_channels, decoder_channels)

        # High-Resolution Final Upsample & Dual-Head Predictor
        self.head_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1),  # Channel 0: Mask, Channel 1: Skeleton
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor (B, in_channels, H, W)

        Returns
        -------
        logits : Tensor (B, 2, H, W)
            Channel 0: Mask Logits
            Channel 1: Centerline / Skeleton Logits
        """
        input_size = (x.shape[-2], x.shape[-1])

        # Multi-scale hierarchical features
        feats = self.encoder(x)
        f2, f3, f4, f5 = feats[0], feats[1], feats[2], feats[3]

        # Top-down feature pyramid fusion
        p5 = self.lat_c5(f5)
        p4 = self.lat_c4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.smooth_c4(p4)

        p3 = self.lat_c3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth_c3(p3)

        p2 = self.lat_c2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth_c2(p2)

        # Upsample fused P2 to native input resolution and apply dual-head predictor
        high_res = F.interpolate(p2, size=input_size, mode="bilinear", align_corners=False)
        logits = self.head_conv(high_res)
        return logits
