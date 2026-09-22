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
    """Lightweight reconstruction decoder for SimMIM with bilinear upsampling to avoid checkerboard artifacts."""

    def __init__(
        self,
        encoder_channels: int = 256,  # Stage 4 output channels for SegFormer B0
        hidden_dim: int = 256,
        output_channels: int = 1,  # Grayscale output
    ) -> None:
        super().__init__()

        # Progressive upsampling from stage 4 (16x16) to full resolution (512x512)
        # Using bilinear upsampling + Conv3x3 to avoid checkerboard artifacts
        # Stage 4: 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
        
        # Stage 4 features: (B, 256, 16, 16) for 512x512 input
        self.up1 = nn.Sequential(
            nn.Conv2d(encoder_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
        )
        
        self.up4 = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 8),
            nn.GELU(),
        )
        
        self.up5 = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 16),
            nn.GELU(),
        )
        
        # Final projection to grayscale
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Ensure output is in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with progressive bilinear upsampling.

        Parameters
        ----------
        x : Tensor (B, C, H, W)
            Encoder feature map from stage 4 (typically 16x16 for 512x512 input)

        Returns
        -------
        reconstructed : Tensor (B, 1, H*32, W*32)
            Reconstructed grayscale image at full resolution
        """
        # Initial projection
        x = self.up1(x)
        
        # Progressive upsampling: 16x16 -> 32x32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(x)
        
        # 32x32 -> 64x64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(x)
        
        # 64x64 -> 128x128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up4(x)
        
        # 128x128 -> 256x256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up5(x)
        
        # 256x256 -> 512x512
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final projection
        x = self.final_conv(x)
        
        return x


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

        # Reconstruction decoder
        self.decoder = SimMIMDecoder(
            encoder_channels=256,  # SegFormer B0 stage 4 channels
            hidden_dim=256,
            output_channels=1,
        )

        # Mask token embedding (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 256, 1, 1))

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
