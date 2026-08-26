"""
Feature Extraction Backbone & FPN Neck for Filament-HQ.

Accepts 4-channel physical inputs (C0: raw, C1: contrast, C2: dark-response, C3: disk distance)
and extracts multi-scale feature pyramids (P2, P3, P4, P5) fused into a unified high-resolution
representation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class BackboneWithFPN4Ch(nn.Module):
    """
    4-channel feature extraction backbone with FPN neck.

    Parameters
    ----------
    model_name : str
        Name of backbone (default 'convnext_tiny').
    in_channels : int
        Number of input image channels (default 4).
    out_channels : int
        Channel width of output FPN feature maps (default 128).
    pretrained : bool
        Load ImageNet pretrained weights for spatial feature filters.
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        in_channels: int = 4,
        out_channels: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.out_channels = out_channels

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for BackboneWithFPN4Ch. Install via: pip install timm")

        # 1. Create backbone with features_only=True
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=in_channels,
            )
        except Exception:
            # Fallback to resnet34 if requested backbone fails
            print(f"[Backbone WARNING] '{model_name}' failed for 4-channel input. Falling back to 'resnet34'.")
            self.backbone = timm.create_model(
                "resnet34",
                pretrained=pretrained,
                features_only=True,
                in_chans=in_channels,
            )

        # Retrieve feature channels for each stage
        dummy = torch.randn(2, in_channels, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy)
        
        self.in_channels_list = [f.shape[1] for f in feats[-4:]]  # P2, P3, P4, P5

        # 2. FPN Lateral & Smooth Convs
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in self.in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in self.in_channels_list
        ])

        # High-resolution fusion layer (combines P2, P3, P4, P5 upsampled to P2 scale)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features and return fused high-resolution feature tensor.

        Parameters
        ----------
        x : Tensor (B, 4, H, W)

        Returns
        -------
        Tensor (B, out_channels, H/4, W/4) — High-res fused P2 feature map.
        """
        feats = self.backbone(x)[-4:]  # [P2, P3, P4, P5]

        # FPN Top-Down Pathway
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        
        # Build top-down feature maps
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[-2:]
            upsampled = F.interpolate(laterals[i], size=target_size, mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply 3x3 smooth convs
        fpn_features = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]
        p2_h, p2_w = fpn_features[0].shape[-2:]

        # Upsample P3, P4, P5 to P2 spatial resolution
        upsampled_fpn = [fpn_features[0]]
        for f in fpn_features[1:]:
            upsampled_fpn.append(F.interpolate(f, size=(p2_h, p2_w), mode="bilinear", align_corners=False))

        # Concatenate and fuse
        fused = torch.cat(upsampled_fpn, dim=1)  # (B, out_channels * 4, P2_H, P2_W)
        p2_fused = self.fusion_conv(fused)       # (B, out_channels, P2_H, P2_W)

        return p2_fused
