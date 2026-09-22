"""
SimMIM (Simple Masked Image Modeling) Pre-training Module for SegFormer B0.

Architecture:
  - Encoder: SegFormer B0 adapted for 1-channel grayscale input
  - Masking: Random block masking (50-60% coverage, 32x32 patch size)
  - Decoder: Lightweight reconstruction decoder from stage 4
  - Output: Full-resolution (512x512) grayscale reconstruction
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SimMIMDecoder(nn.Module):
    """Lightweight reconstruction decoder for SimMIM."""

    def __init__(
        self,
        encoder_channels: int = 256,  # Stage 4 output channels for SegFormer B0
        hidden_dim: int = 256,
        output_channels: int = 1,  # Grayscale output
    ) -> None:
        super().__init__()

        # Progressive upsampling from stage 4 to full resolution
        # Stage 4: stride 32 -> 16 -> 8 -> 4 -> 2 -> 1 (for 512x512 input)
        self.decoder = nn.Sequential(
            # Stage 4 to Stage 3 (stride 32 -> 16)
            nn.Conv2d(encoder_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),

            # Stage 3 to Stage 2 (stride 16 -> 8)
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),

            # Stage 2 to Stage 1 (stride 8 -> 4)
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 8),
            nn.GELU(),

            # Stage 1 to stride 2 (stride 4 -> 2)
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 8),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 16),
            nn.GELU(),

            # Stride 2 to full resolution (stride 2 -> 1)
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 16),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim // 16, hidden_dim // 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 32),
            nn.GELU(),

            # Final projection to grayscale
            nn.Conv2d(hidden_dim // 32, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Encoder feature map from stage 4

        Returns
        -------
        reconstructed : Tensor (B, 1, H*32, W*32)
            Reconstructed grayscale image at full resolution
        """
        return self.decoder(x)


class SimMIMSegFormer(nn.Module):
    """
    SimMIM model with SegFormer B0 encoder.

    Parameters
    ----------
    backbone_name : str
        SegFormer model name (default: 'nvidia/mit-b0')
    in_channels : int
        Input channels (default: 1 for grayscale)
    mask_ratio : float
        Masking ratio (default: 0.5 for 50% coverage)
    patch_size : int
        Patch size for masking (default: 32)
    pretrained : bool
        Whether to use pretrained weights (default: False for SimMIM)
    """

    def __init__(
        self,
        backbone_name: str = "nvidia/mit-b0",
        in_channels: int = 1,
        mask_ratio: float = 0.5,
        patch_size: int = 32,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

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

        # Reconstruction decoder
        self.decoder = SimMIMDecoder(
            encoder_channels=256,  # SegFormer B0 stage 4 channels
            hidden_dim=256,
            output_channels=1,
        )

        # Mask token embedding (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 256, 1, 1))

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random block mask.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Input image
        mask_ratio : float
            Masking ratio

        Returns
        -------
        mask : Tensor (B, 1, H, W)
            Binary mask (1 = masked, 0 = visible)
        """
        B, C, H, W = x.shape
        patch_size = self.patch_size

        # Calculate number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w

        # Generate random mask
        mask = torch.rand(B, num_patches, device=x.device) < mask_ratio

        # Reshape to spatial
        mask = mask.view(B, num_patches_h, num_patches_w)
        mask = mask.unsqueeze(1).float()

        # Upsample to full resolution
        mask = F.interpolate(
            mask,
            size=(H, W),
            mode='nearest'
        )

        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor (B, 1, H, W)
            Input grayscale image
        mask : Tensor (B, 1, H, W) or None
            Pre-computed mask (if None, random mask is generated)

        Returns
        -------
        dict containing:
            - 'reconstructed': Reconstructed image (B, 1, H, W)
            - 'mask': Binary mask (B, 1, H, W)
            - 'encoder_features': Stage 4 features (B, 256, H/32, W/32)
        """
        B, C, H, W = x.shape

        # Generate random mask if not provided
        if mask is None:
            mask = self.random_masking(x, self.mask_ratio)

        # Apply mask to input (replace masked regions with mask token)
        masked_x = x * (1 - mask)

        # Forward through encoder
        outputs = self.encoder.segformer(masked_x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Get stage 4 features (deepest)
        stage4_features = hidden_states[3]  # (B, 256, H/32, W/32)

        # Upsample mask to feature resolution
        mask_downsampled = F.interpolate(
            mask,
            size=stage4_features.shape[-2:],
            mode='nearest'
        )

        # Replace masked regions with mask token
        stage4_features_masked = stage4_features * (1 - mask_downsampled) + \
                                 self.mask_token * mask_downsampled

        # Reconstruct through decoder
        reconstructed = self.decoder(stage4_features_masked)

        return {
            'reconstructed': reconstructed,
            'mask': mask,
            'encoder_features': stage4_features,
        }

    def get_encoder(self) -> nn.Module:
        """Return the encoder for transfer learning."""
        return self.encoder.segformer
