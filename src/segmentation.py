"""
Supervised Segmentation Module with Pre-trained SimMIM Encoder.

Architecture:
  - Encoder: SegFormer B0 (can be initialized from SimMIM pre-training)
  - Decoder: Multi-scale Feature Pyramid Decoder
  - Output: Dual-channel logits (mask + skeleton)
"""

from __future__ import annotations

from typing import Optional

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


class SolarFilamentSegmentation(nn.Module):
    """
    Supervised segmentation model with dual-channel output.

    Parameters
    ----------
    backbone_name : str
        SegFormer model name (default: 'nvidia/mit-b0')
    in_channels : int
        Input channels (default: 1 for grayscale)
    decoder_channels : int
        Decoder intermediate channels (default: 128)
    pretrained : bool
        Whether to use ImageNet pretrained weights (default: True)
    simmim_checkpoint : str or None
        Path to SimMIM pre-trained encoder checkpoint
    """

    def __init__(
        self,
        backbone_name: str = "nvidia/mit-b0",
        in_channels: int = 1,
        decoder_channels: int = 128,
        pretrained: bool = True,
        simmim_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels = in_channels

        # Instantiate SegFormer encoder
        if pretrained:
            self.encoder = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,
                ignore_mismatched_sizes=True
            )
        else:
            self.encoder = SegformerForSemanticSegmentation.from_pretrained(
                backbone_name,
                num_labels=1,
                ignore_mismatched_sizes=True
            )

        # Modify first layer to accept 1-channel input
        if in_channels == 1:
            original_proj = self.encoder.segformer.stages[0].patch_embeddings.proj
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

            self.encoder.segformer.stages[0].patch_embeddings.proj = new_proj

        # Load SimMIM pre-trained weights if provided
        if simmim_checkpoint is not None:
            self.load_simmim_weights(simmim_checkpoint)

        # Multi-scale Feature Pyramid Decoder
        # SegFormer B0 channels: [32, 64, 160, 256]
        enc_channels = [32, 64, 160, 256]
        c2, c3, c4, c5 = enc_channels

        # Lateral 1x1 projections
        self.lat_c5 = nn.Conv2d(c5, decoder_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4, decoder_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.lat_c2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)

        # Smooth fusion blocks
        self.smooth_c4 = ConvBlock(decoder_channels, decoder_channels)
        self.smooth_c3 = ConvBlock(decoder_channels, decoder_channels)
        self.smooth_c2 = ConvBlock(decoder_channels, decoder_channels)

        # Dual-head predictor
        self.head_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1),  # Channel 0: Mask, Channel 1: Skeleton
        )

    def load_simmim_weights(self, checkpoint_path: str) -> None:
        """
        Load pre-trained encoder weights from SimMIM checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to SimMIM checkpoint (.pt file)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract encoder state dict
        if 'encoder' in checkpoint:
            encoder_state = checkpoint['encoder']
        elif 'model' in checkpoint:
            encoder_state = checkpoint['model']
        else:
            encoder_state = checkpoint

        # Load encoder weights (skip mismatched keys)
        current_state = self.encoder.segformer.state_dict()
        matched_keys = []

        for name, param in encoder_state.items():
            if name in current_state and param.shape == current_state[name].shape:
                current_state[name] = param
                matched_keys.append(name)

        self.encoder.segformer.load_state_dict(current_state, strict=False)
        print(f"Loaded {len(matched_keys)} encoder parameters from SimMIM checkpoint")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor (B, 1, H, W)
            Input grayscale image

        Returns
        -------
        logits : Tensor (B, 2, H, W)
            Channel 0: Mask logits
            Channel 1: Skeleton logits
        """
        input_size = (x.shape[-2], x.shape[-1])

        # Multi-scale features from encoder
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

        # Upsample to full resolution
        high_res = F.interpolate(p2, size=input_size, mode="bilinear", align_corners=False)
        logits = self.head_conv(high_res)

        return logits
