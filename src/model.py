"""
Ultra-Efficient, Lightweight Neural Architecture for Solar Filament Micro-Segmentation.

Architecture:
  - Backbone: SegFormer B0 from transformers (Mix Transformer for SimMIM)
  - Decoder: Lightweight multi-scale Feature Pyramid Decoder (128 channels)
  - Output: 2-Channel High-Resolution Logits:
      Channel 0: Filament Mask Logits
      Channel 1: Filament Topological Skeleton Logits
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


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
    Lightweight, High-Precision Micro-Segmentation Network with SegFormer B0.

    Parameters
    ----------
    backbone_name : str
        Encoder model (default: 'nvidia/mit-b0' for SegFormer B0).
    in_channels : int
        Number of input image channels (default: 1 for grayscale SimMIM).
    decoder_channels : int
        Intermediate feature channels in the decoder (default: 128).
    pretrained : bool
        Whether to initialize encoder with pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str = "nvidia/mit-b0",
        in_channels: int = 1,
        decoder_channels: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels = in_channels

        # Instantiate SegFormer from transformers
        if pretrained:
            self.encoder = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,  # Will be overridden by custom head
                ignore_mismatched_sizes=True
            )
        else:
            self.encoder = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,
                ignore_mismatched_sizes=True
            )

        # Modify first layer to accept 1-channel input (grayscale for SimMIM)
        if in_channels == 1:
            # Handle different transformers versions
            if hasattr(self.encoder.segformer, 'stages'):
                # Older version
                original_proj = self.encoder.segformer.stages[0].patch_embeddings.proj
                self.encoder.segformer.stages[0].patch_embeddings.proj = self._create_1ch_conv(original_proj, in_channels)
            elif hasattr(self.encoder.segformer, 'encoder'):
                # Newer version with nested encoder
                if hasattr(self.encoder.segformer.encoder, 'patch_embeddings'):
                    patch_embeddings = self.encoder.segformer.encoder.patch_embeddings
                    if isinstance(patch_embeddings, nn.ModuleList):
                        # ModuleList case - access first element
                        original_proj = patch_embeddings[0].proj
                        patch_embeddings[0].proj = self._create_1ch_conv(original_proj, in_channels)
                    else:
                        # Single module case
                        original_proj = patch_embeddings.proj
                        patch_embeddings.proj = self._create_1ch_conv(original_proj, in_channels)
                elif hasattr(self.encoder.segformer.encoder, 'embeddings'):
                    # Even newer version
                    patch_embeddings = self.encoder.segformer.encoder.embeddings.patch_embeddings
                    if isinstance(patch_embeddings, nn.ModuleList):
                        original_proj = patch_embeddings[0].proj
                        patch_embeddings[0].proj = self._create_1ch_conv(original_proj, in_channels)
                    else:
                        original_proj = patch_embeddings.proj
                        patch_embeddings.proj = self._create_1ch_conv(original_proj, in_channels)
            else:
                # Try to find the first conv layer
                for name, module in self.encoder.segformer.named_modules():
                    if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                        print(f"Found first conv layer: {name}")
                        parent_name = '.'.join(name.split('.')[:-1])
                        parent = self.encoder.segformer
                        for part in parent_name.split('.'):
                            parent = getattr(parent, part)
                        last_name = name.split('.')[-1]
                        setattr(parent, last_name, self._create_1ch_conv(module, in_channels))
                        break

        # Get encoder channels: SegFormer B0 has [32, 64, 160, 256]
        enc_channels = [32, 64, 160, 256]
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

    def _create_1ch_conv(self, original_proj: nn.Conv2d, in_channels: int) -> nn.Conv2d:
        """Create 1-channel conv from 3-channel conv."""
        original_weight = original_proj.weight
        original_out_channels = original_weight.shape[0]

        new_proj = nn.Conv2d(
            in_channels,
            original_out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None
        )

        with torch.no_grad():
            new_proj.weight = nn.Parameter(
                original_weight.mean(dim=1, keepdim=True)
            )
        if original_proj.bias is not None:
            new_proj.bias = nn.Parameter(original_proj.bias)

        return new_proj

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

        # Get multi-scale features from SegFormer encoder
        # SegFormer outputs 4 feature maps with channels [32, 64, 160, 256]
        outputs = self.encoder.segformer(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        f2, f3, f4, f5 = hidden_states[0], hidden_states[1], hidden_states[2], hidden_states[3]

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
