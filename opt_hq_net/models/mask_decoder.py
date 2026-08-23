"""
HQ-SAM inspired Mask Decoder with High-Quality (HQ) token.

Architecture summary
--------------------
1. ``PromptEncoder``
   Converts oriented bounding box parameters (xc, yc, w, h, θ) into
   dense positional embeddings that are concatenated with the RoI crop
   feature before decoding.

2. ``HQMaskDecoder``
   A 2-layer transformer decoder that:
   a) Maintains a learnable set of instance queries + one HQ token.
   b) Applies masked cross-attention to the 28×28 RoI crop features.
   c) Fuses the HQ token with the high-resolution P2 feature map
      (sub-pixel detail preservation for filament barbs).
   d) Predicts a 28×28 binary mask probability map per instance.
   e) Upscales to the target resolution via bilinear interpolation.

This module deliberately avoids the full SAM ViT encoder overhead —
the backbone already computes a rich multi-scale feature pyramid.
The decoder is lightweight: ~4M parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_hq_net.config import MaskDecoderConfig


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """
    Sine/cosine positional embedding following DETR (Carion et al. 2020).

    Parameters
    ----------
    num_pos_feats : int
        Number of positional features (half of the full embedding dim).
    temperature : float
        Temperature scaling for frequency computation.
    """

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (B, C, H, W)

        Returns
        -------
        Tensor (B, num_pos_feats*2, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device)

        # Normalise to [0, 2π]
        y_embed = y_embed / H * 2 * math.pi
        x_embed = x_embed / W * 2 * math.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t[None, :]   # (W, D)
        pos_y = y_embed[:, None] / dim_t[None, :]   # (H, D)

        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([
            pos_y.unsqueeze(1).expand(-1, W, -1),  # (H, W, D)
            pos_x.unsqueeze(0).expand(H, -1, -1),  # (H, W, D)
        ], dim=-1).permute(2, 0, 1).unsqueeze(0)   # (1, 2D, H, W)

        return pos.expand(B, -1, -1, -1)


# ---------------------------------------------------------------------------
# Prompt encoder
# ---------------------------------------------------------------------------

class PromptEncoder(nn.Module):
    """
    Encode oriented bounding box parameters as dense embeddings.

    Produces a (1, embed_dim) prompt token per box that is appended to
    the transformer input sequence.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimensionality (matches decoder hidden dim).
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projection from 5D box params to embed_dim
        self.box_proj = nn.Sequential(
            nn.Linear(5, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Learnable orientation embedding (captures θ information explicitly)
        self.angle_embed = nn.Linear(2, embed_dim, bias=False)  # (sin θ, cos θ) → D

    # ------------------------------------------------------------------
    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        boxes : Tensor (N, 5)
            Oriented boxes [xc, yc, w, h, θ_rad].

        Returns
        -------
        Tensor (N, embed_dim)
        """
        theta = boxes[:, 4]
        angle_feat = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        return self.box_proj(boxes) + self.angle_embed(angle_feat)


# ---------------------------------------------------------------------------
# Transformer decoder block
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer (self-attention → cross-attention → FFN).

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.
    dim_feedforward : int
        FFN hidden dimensionality.
    dropout : float
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout(q2))
        # Cross-attention with image features
        q2, _ = self.cross_attn(queries, keys, values)
        queries = self.norm2(queries + self.dropout(q2))
        # FFN
        queries = self.norm3(queries + self.dropout(self.ffn(queries)))
        return queries


# ---------------------------------------------------------------------------
# HQ Mask Decoder
# ---------------------------------------------------------------------------

class HQMaskDecoder(nn.Module):
    """
    High-Quality Mask Decoder inspired by HQ-SAM.

    Inputs
    ------
    roi_crops    : Tensor (N_total, C, 28, 28)
        Feature crops extracted by Rotated RoIAlign.
    prompt_embs  : Tensor (N_total, D)
        Prompt embeddings from ``PromptEncoder``.
    p2_features  : Tensor (B, C, H/4, W/4)
        Full P2 feature map for HQ token fusion.
    batch_idx    : Tensor (N_total,)
        Batch membership of each crop.
    boxes        : list[Tensor]
        Oriented boxes per image — used to localise the HQ token.

    Output
    ------
    masks   : Tensor (N_total, 1, H_out, W_out) — predicted probability maps
    """

    def __init__(self, cfg: MaskDecoderConfig) -> None:
        super().__init__()
        D = cfg.transformer_dim

        # Project RoI crop features to decoder dim
        self.crop_proj = nn.Sequential(
            nn.Conv2d(cfg.p2_channels, D, kernel_size=1),
            nn.GELU(),
        )

        # Positional encoding for 28×28 grid
        self.pos_enc = PositionEmbeddingSine(num_pos_feats=D // 2)

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(embed_dim=D)

        # HQ token — learnable embedding fused with P2 high-res features
        self.hq_token = nn.Parameter(torch.zeros(1, D))
        nn.init.normal_(self.hq_token, std=0.02)

        # P2 → D projection for the HQ feature fusion
        self.hq_proj = nn.Sequential(
            nn.Conv2d(cfg.p2_channels, D, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(D, D, kernel_size=1),
        )

        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=D,
                num_heads=cfg.num_heads,
                dim_feedforward=cfg.mlp_dim,
            )
            for _ in range(cfg.num_layers)
        ])

        # Output upscaling + mask head
        self.mask_upscale = nn.Sequential(
            nn.ConvTranspose2d(D, D // 2, kernel_size=2, stride=2),   # 28 → 56
            nn.GELU(),
            nn.ConvTranspose2d(D // 2, D // 4, kernel_size=2, stride=2),  # 56 → 112
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(D // 4, 1, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(
        self,
        roi_crops: torch.Tensor,          # (N, C, 28, 28)
        boxes_per_image: List[torch.Tensor],  # each (N_i, 5+)
        p2_features: torch.Tensor,        # (B, C, H/4, W/4)
        batch_idx: torch.Tensor,          # (N,)
    ) -> torch.Tensor:
        """
        Decode instance masks.

        Returns
        -------
        Tensor (N, 1, 112, 112) — predicted mask logits (before sigmoid).
        """
        N = roi_crops.shape[0]
        if N == 0:
            return roi_crops.new_zeros(0, 1, 112, 112)

        device = roi_crops.device

        # Project crops: (N, D, 28, 28)
        features_2d = self.crop_proj(roi_crops)

        # Positional encoding: (N, D, 28, 28)
        pos = self.pos_enc(features_2d)

        # Flatten to sequence: (N, 28*28, D)
        N_crops, D, H_c, W_c = features_2d.shape
        feat_seq = (features_2d + pos).flatten(2).permute(0, 2, 1)  # (N, HW, D)

        # Build prompt queries: (N, D)
        all_boxes = torch.cat(
            [boxes[:, :5] for boxes in boxes_per_image if len(boxes) > 0],
            dim=0,
        )  # (N, 5)
        prompt_embs = self.prompt_encoder(all_boxes)   # (N, D)

        # HQ token fused with P2 features (per crop, using box centre)
        hq_token = self.hq_token.expand(N, -1)         # (N, D)

        # Compute HQ features from P2 at the box location
        hq_feat = self._extract_hq_features(
            p2_features, boxes_per_image, batch_idx
        )  # (N, D)
        hq_token = hq_token + hq_feat

        # Query sequence: [prompt_token, hq_token] → (N, 2, D)
        queries = torch.stack([prompt_embs, hq_token], dim=1)

        # Run transformer decoder layers in chunks of 64 to prevent MultiheadAttention OOM
        if N > 64:
            queries_out = []
            chunk_size = 64
            for start in range(0, N, chunk_size):
                q_chunk = queries[start:start + chunk_size]
                f_chunk = feat_seq[start:start + chunk_size]
                for layer in self.decoder_layers:
                    q_chunk = layer(q_chunk, f_chunk, f_chunk)
                queries_out.append(q_chunk)
            queries = torch.cat(queries_out, dim=0)
        else:
            for layer in self.decoder_layers:
                queries = layer(queries, feat_seq, feat_seq)

        # Use HQ token output (index 1) for the final mask prediction
        hq_out = queries[:, 1, :]   # (N, D)

        # Reshape to 2D and decode
        hq_2d = hq_out.unsqueeze(-1).unsqueeze(-1) * features_2d  # (N, D, 28, 28)
        upscaled = self.mask_upscale(hq_2d)                        # (N, D/4, 112, 112)
        masks = self.mask_head(upscaled)                            # (N, 1, 112, 112)

        return masks

    # ------------------------------------------------------------------
    def _extract_hq_features(
        self,
        p2_features: torch.Tensor,
        boxes_per_image: List[torch.Tensor],
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample P2 features at each box centre to form the HQ context vector.

        Parameters
        ----------
        p2_features : (B, C, H_p2, W_p2)
        boxes_per_image : list[Tensor (N_i, 5+)]

        Returns
        -------
        Tensor (N_total, D) — HQ context from P2
        """
        B, C, H_p, W_p = p2_features.shape
        device = p2_features.device

        # Project P2 to D
        p2_proj = self.hq_proj(p2_features)   # (B, D, H_p, W_p)

        all_contexts: List[torch.Tensor] = []
        for img_idx, boxes in enumerate(boxes_per_image):
            if len(boxes) == 0:
                continue
            xc_n = boxes[:, 0] / (W_p * 4) * 2 - 1   # image space → normalised
            yc_n = boxes[:, 1] / (H_p * 4) * 2 - 1
            grid = torch.stack([xc_n, yc_n], dim=-1).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
            p2_i = p2_proj[img_idx].unsqueeze(0)                                # (1, D, H_p, W_p)
            sampled = F.grid_sample(p2_i, grid, mode="bilinear", align_corners=False) # (1, D, N, 1)
            all_contexts.append(sampled.squeeze(0).squeeze(-1).permute(1, 0))        # (N, D)

        if not all_contexts:
            D = p2_proj.shape[1]
            return torch.zeros(0, D, device=device)

        return torch.cat(all_contexts, dim=0)   # (N_total, D)
