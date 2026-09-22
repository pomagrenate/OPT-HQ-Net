"""
MicroFilNet -- an ultra-compact, topology-aware segmentation network for
solar filaments (MAGFiLO / GONG H-alpha).

Design goals and where the parameter budget goes (see README.md Section 2
for the full rationale):

  1. FLAT channel economy  : channels grow gently (40 -> 64 -> 96 -> 128)
     instead of the classic U-Net doubling (64->128->256->512), which is
     the single largest source of bloat in a standard U-Net.
  2. DEPTHWISE-SEPARABLE convolutions everywhere instead of full 3x3 convs
     (~8-9x fewer params per conv layer, MobileNet-style).
  3. SCA -- Simplified Channel Attention: a 2-layer squeeze/excite gate,
     cheap channel-wise recalibration instead of extra conv depth.
  4. Global context WITHOUT quadratic attention: a stack of dilated
     depthwise-separable residual blocks (dilations 1,2,4,8,1,2,4,8) at the
     bottleneck gives a very large effective receptive field for a tiny
     param cost (dilation is free -- it doesn't change kernel size).
  5. Edge/ridge-guided FiLM modulation: the zero-parameter ridge prior
     computed in preprocessing.py is projected (2*C params) into a
     per-channel scale+shift applied at the bottleneck -- this is the
     parameter-cheap analogue of EdgeAttNet's edge-guided attention bias,
     costing a few hundred parameters instead of extra attention heads.
  6. Two lightweight linear-attention blocks (O(N) cost, Shen et al. 2021
     formulation) at the bottleneck for genuine long-range mixing -- this
     is what lets a same-magnitude dark patch far away on the disk still
     get grouped into the same filament instance, directly targeting the
     "structural continuity" challenge.

Total parameter count for the default config: ~705K (verified by hand
parameter-accounting in param_calc2.py, mirrored 1:1 by the layer shapes
below; run `python model.py` to get the exact count via
`sum(p.numel() for p in model.parameters())`).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

class DWSepConv(nn.Module):
    """Depthwise 3x3 (optionally dilated) + pointwise 1x1, each with BN+ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=pad, dilation=dilation,
                             groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class SCA(nn.Module):
    """Simplified Channel Attention: global-avg-pool -> FC -> FC -> sigmoid gate."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hid = max(4, ch // reduction)
        self.fc1 = nn.Linear(ch, hid, bias=False)
        self.fc2 = nn.Linear(hid, ch, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 3):
        super().__init__()
        blocks = [DWSepConv(in_ch, out_ch)]
        for _ in range(n_blocks - 1):
            blocks.append(DWSepConv(out_ch, out_ch))
        self.blocks = nn.Sequential(*blocks)
        self.sca = SCA(out_ch)

    def forward(self, x):
        x = self.blocks(x)
        x = self.sca(x)
        skip = x
        pooled = F.max_pool2d(x, 2)
        return pooled, skip


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, n_blocks: int = 3):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
        refine = [DWSepConv(out_ch, out_ch) for _ in range(n_blocks - 1)]
        self.refine = nn.Sequential(*refine) if refine else nn.Identity()
        self.sca = SCA(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.refine(x)
        x = self.sca(x)
        return x


class RidgeFiLM(nn.Module):
    """
    Projects the single-channel zero-parameter ridge/continuity prior
    (from preprocessing.py) into a per-channel scale (gamma) and shift
    (beta) applied to the bottleneck features -- a cheap alternative to
    concatenation that costs only 2*C parameters and lets the classical
    prior directly steer which channels the dilated block emphasizes.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.proj = nn.Conv2d(1, 2 * ch, 1)  # -> (gamma, beta)
        self.ch = ch

    def forward(self, x, ridge_map):
        ridge_small = F.interpolate(ridge_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gb = self.proj(ridge_small)
        gamma, beta = gb[:, :self.ch], gb[:, self.ch:]
        return x * (1 + torch.tanh(gamma)) + beta


class LinearAttention2D(nn.Module):
    """
    O(N) "efficient attention" (Shen et al., 2021): softmax(Q) over the
    channel dim times softmax(K) over the spatial dim, avoiding the
    N x N attention matrix altogether. Parameter cost is only the
    Q/K/V/Out 1x1 projections (4*dim^2), no positional encodings needed.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q = q.reshape(b, self.heads, self.dh, h * w)
        k = k.reshape(b, self.heads, self.dh, h * w)
        v = v.reshape(b, self.heads, self.dh, h * w)
        q = q.softmax(dim=2)     # normalize over feature dim
        k = k.softmax(dim=-1)    # normalize over spatial dim
        context = torch.einsum("bhdn,bhen->bhde", k, v)   # (b, heads, dh, dh)
        out = torch.einsum("bhde,bhdn->bhen", context, q)  # (b, heads, dh, N)
        out = out.reshape(b, c, h, w)
        return self.to_out(out)


class DilatedBottleneck(nn.Module):
    """
    Global-context block: 8 dilated depthwise-separable residual convs
    (dilation pattern 1,2,4,8 repeated twice) followed by ridge-FiLM
    modulation and two linear-attention passes. Dilation grows the
    receptive field for free (no extra params vs. dilation=1), and is
    the main mechanism that lets the network reason about whether two
    distant dark blobs are part of the same physical filament.
    """

    def __init__(self, ch: int, n_dilated_pairs: int = 4):
        super().__init__()
        dilations = [1, 2, 4, 8] * (n_dilated_pairs // 4 + 1)
        dilations = dilations[:n_dilated_pairs]
        self.blocks = nn.ModuleList([DWSepConv(ch, ch, dilation=d) for d in dilations])
        self.film = RidgeFiLM(ch)
        self.attn1 = LinearAttention2D(ch)
        self.attn2 = LinearAttention2D(ch)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x, ridge_map):
        for blk in self.blocks:
            x = x + blk(x)
        x = self.film(x, ridge_map)
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x))
        return x


# --------------------------------------------------------------------------
# MicroFilNet
# --------------------------------------------------------------------------

class MicroFilNet(nn.Module):
    def __init__(self,
                 in_ch: int = 2,               # [enhanced H-alpha, ridge prior]
                 stem_ch: int = 32,
                 widths=(40, 64, 96, 128),
                 bottleneck_ch: int = 160,
                 n_blocks_per_stage: int = 3,
                 n_dilated: int = 8):
        super().__init__()
        self.stem = DWSepConv(in_ch, stem_ch)

        enc_in = [stem_ch] + list(widths[:-1])
        self.encoders = nn.ModuleList([
            EncoderStage(ic, oc, n_blocks_per_stage) for ic, oc in zip(enc_in, widths)
        ])

        self.bottleneck_in = nn.Sequential(
            nn.Conv2d(widths[-1], bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU6(inplace=True),
        )
        self.bottleneck = DilatedBottleneck(bottleneck_ch, n_dilated)

        dec_widths = list(reversed(widths))
        dec_in = [bottleneck_ch] + dec_widths[:-1]
        self.decoders = nn.ModuleList([
            DecoderStage(ic, sc, sc, n_blocks_per_stage)
            for ic, sc in zip(dec_in, dec_widths)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(widths[0], 8, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        """
        x: (B, 2, H, W) -- channel 0 = preprocessed H-alpha, channel 1 = ridge prior.
        Returns raw logits (B, 1, H, W); apply sigmoid outside for probabilities.
        """
        ridge_map = x[:, 1:2]  # keep the full-res ridge prior for FiLM

        feat = self.stem(x)
        skips = []
        for enc in self.encoders:
            feat, skip = enc(feat)
            skips.append(skip)

        feat = self.bottleneck_in(feat)
        feat = self.bottleneck(feat, ridge_map)

        for dec, skip in zip(self.decoders, reversed(skips)):
            feat = dec(feat, skip)

        logits = self.head(feat)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MicroFilNet()
    n = count_parameters(model)
    print(f"MicroFilNet trainable parameters: {n:,}")
    x = torch.randn(1, 2, 256, 256)
    y = model(x)
    print(f"Output shape: {tuple(y.shape)}")

    # Per-block breakdown, useful for re-tuning the width sweep
    print("\n--- parameter breakdown ---")
    for name, module in model.named_children():
        n_sub = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:15s}: {n_sub:>10,}")
