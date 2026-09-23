from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWSepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, padding=dilation, dilation=dilation, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class SCA(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hid = max(4, ch // reduction)
        self.fc1 = nn.Linear(ch, hid, bias=False)
        self.fc2 = nn.Linear(hid, ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.sca(self.blocks(x))
        return F.max_pool2d(x, 2), x


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

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([x, skip], dim=1))
        return self.sca(self.refine(x))


class LinearAttention2D(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.dh = dim // heads
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        if edge_bias is not None:
            q = q + edge_bias
            k = k + edge_bias

        q = q.reshape(b, self.heads, self.dh, h * w).softmax(dim=2)
        k = k.reshape(b, self.heads, self.dh, h * w).softmax(dim=-1)
        v = v.reshape(b, self.heads, self.dh, h * w)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q).reshape(b, c, h, w)
        return self.to_out(out)


class TinyGlobalEncoder(nn.Module):
    """Encodes the whole-disk (downsampled) image into a small feature MAP.

    Previously this ended in AdaptiveAvgPool2d((1, 1)), collapsing the entire
    disk into a single vector that was broadcast identically to every tile.
    That threw away exactly the information a tile needs to stay consistent
    with its neighbours (e.g. a filament that continues past the tile edge).
    We now keep the spatial map and let the caller sample the region that
    matters for a given tile (see `spatial_align_crop` below).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DWSepConv(32, 64),
            nn.MaxPool2d(2),
            DWSepConv(64, 96),
            nn.MaxPool2d(2),
            DWSepConv(96, out_ch),
            # No AdaptiveAvgPool2d here on purpose: keep the spatial map.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_ch, Hg, Wg)


def spatial_align_crop(
    feat_map: torch.Tensor,
    bbox_norm: torch.Tensor,
    out_size: tuple[int, int],
) -> torch.Tensor:
    """Crop-and-resize `feat_map` to the region given by `bbox_norm`, per batch item.

    Args:
        feat_map: (B, C, Hg, Wg) feature map covering the FULL frame that the
            global image was resized from (i.e. the whole solar disk image).
        bbox_norm: (B, 4) tensor of (x0, y0, x1, y1), each a fraction of the
            full frame's width/height. Values may fall outside [0, 1] (e.g.
            a tile's context box padded past the image border); those are
            handled with border padding rather than clipped beforehand, so
            near-limb tiles still get a sensible (edge-replicated) context.
        out_size: (H, W) spatial size to resample the crop to — typically the
            spatial size of the local feature map it will be fused with.

    Returns:
        (B, C, out_size[0], out_size[1]) tensor: for each batch item, the
        `feat_map` region described by its bbox, resampled to `out_size`.
    """
    b = feat_map.shape[0]
    x0 = bbox_norm[:, 0] * 2.0 - 1.0
    y0 = bbox_norm[:, 1] * 2.0 - 1.0
    x1 = bbox_norm[:, 2] * 2.0 - 1.0
    y1 = bbox_norm[:, 3] * 2.0 - 1.0

    a_x = (x1 - x0) / 2.0
    b_x = (x0 + x1) / 2.0
    a_y = (y1 - y0) / 2.0
    b_y = (y0 + y1) / 2.0

    theta = torch.zeros(b, 2, 3, dtype=feat_map.dtype, device=feat_map.device)
    theta[:, 0, 0] = a_x
    theta[:, 0, 2] = b_x
    theta[:, 1, 1] = a_y
    theta[:, 1, 2] = b_y

    grid = F.affine_grid(
        theta, size=(b, feat_map.shape[1], out_size[0], out_size[1]), align_corners=False
    )
    return F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=False)


class StripPooling(nn.Module):
    """Horizontal + vertical strip pooling (Hou et al., "Strip Pooling:
    Rethinking Spatial Pooling for Scene Parsing", CVPR 2020).

    A filament is a thin, elongated curve that can run most of the way
    across a tile. Square convolution kernels (even dilated ones) see very
    little of a filament's own length in one pass, so the network has to
    stack many layers just to link up two ends of the same thin structure.
    Strip pooling collapses the feature map along one axis at a time
    (H -> a length-H column, W -> a length-W row), so a single 1x1 conv
    already mixes information along the *entire* row/column a filament
    might occupy, then broadcasts that context back out. It's a handful of
    1x1 convs, so the parameter cost is tiny relative to the rest of the
    model.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.conv_h = nn.Conv2d(ch, ch, 1, bias=False)
        self.conv_v = nn.Conv2d(ch, ch, 1, bias=False)
        self.fuse = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_h = self.conv_h(F.adaptive_avg_pool2d(x, (h, 1)))  # (B, C, H, 1)
        x_v = self.conv_v(F.adaptive_avg_pool2d(x, (1, w)))  # (B, C, 1, W)
        strip = self.fuse(x_h + x_v)  # broadcasts to (B, C, H, W)
        gate = torch.sigmoid(self.bn(strip))
        return x + x * gate


class DilatedBottleneck(nn.Module):
    def __init__(self, ch: int, n_dilated_pairs: int = 4):
        super().__init__()
        dilations = [1, 2, 4, 8] * (n_dilated_pairs // 4 + 1)
        dilations = dilations[:n_dilated_pairs]
        self.blocks = nn.ModuleList([DWSepConv(ch, ch, dilation=d) for d in dilations])
        self.strip_pool = StripPooling(ch)
        self.edge_proj = nn.Conv2d(1, ch, 1, bias=False)
        self.attn1 = LinearAttention2D(ch)
        self.attn2 = LinearAttention2D(ch)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x: torch.Tensor, edge_map: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = x + blk(x)

        x = self.strip_pool(x)

        edge_small = F.interpolate(edge_map, size=x.shape[-2:], mode="bilinear", align_corners=False)
        edge_bias = self.edge_proj(edge_small)

        x = x + self.attn1(self.norm1(x), edge_bias)
        x = x + self.attn2(self.norm2(x), edge_bias)
        return x


class MicroFilNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        stem_ch: int = 32,
        widths: tuple[int, ...] = (40, 64, 96, 128),
        bottleneck_ch: int = 160,
        global_feat_ch: int = 128,
        coord_dim: int = 3,
        n_blocks_per_stage: int = 3,
        n_dilated: int = 8,
    ):
        super().__init__()
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.stem = DWSepConv(in_ch, stem_ch)

        enc_in = [stem_ch] + list(widths[:-1])
        self.encoders = nn.ModuleList([
            EncoderStage(ic, oc, n_blocks_per_stage) for ic, oc in zip(enc_in, widths)
        ])

        self.global_encoder = TinyGlobalEncoder(in_ch=in_ch, out_ch=global_feat_ch)
        self.coord_proj = nn.Linear(coord_dim, 32)

        fusion_in_ch = widths[-1] + global_feat_ch + 32
        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(fusion_in_ch, bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU6(inplace=True),
        )
        self.bottleneck = DilatedBottleneck(bottleneck_ch, n_dilated)

        dec_widths = list(reversed(widths))
        dec_in = [bottleneck_ch] + dec_widths[:-1]
        self.decoders = nn.ModuleList([
            DecoderStage(ic, sc, sc, n_blocks_per_stage) for ic, oc, sc in zip(dec_in, dec_widths, dec_widths)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(widths[0], 8, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        # Filament pixels are a tiny fraction of a tile (well under 1%).
        # With the default zero bias, an untrained head outputs sigmoid(x)
        # near 0.5 everywhere, i.e. it starts by predicting most of the tile
        # (or the whole disk) as filament — exactly the "predicts the whole
        # sun" behavior seen in early epochs. Starting the bias low makes
        # the untrained network's prior match the true class balance, so
        # early training doesn't have to fight its way down from "everything
        # is foreground" before it can start learning where filaments
        # actually are.
        nn.init.constant_(self.head[-1].bias, -4.0)

    def forward(
        self,
        local_x: torch.Tensor,
        global_x: torch.Tensor,
        coords: torch.Tensor,
        bbox_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            local_x: (B, in_ch, tile, tile) local tile.
            global_x: (B, in_ch, Gh, Gw) whole-disk image, downsampled.
            coords: (B, 3) = (x_norm, y_norm, r_norm) of the tile CENTER,
                same as before — still useful as an absolute-position cue.
            bbox_norm: (B, 4) = (x0, y0, x1, y1) of the tile's CONTEXT WINDOW
                (the tile extent, typically padded with a margin) as a
                fraction of the full-disk frame. This is what lets the model
                see structure beyond its own tile boundary. See dataset.py /
                inference.py for how this is computed.
        """
        edge_map = self.edge_extractor(local_x)

        feat = self.stem(local_x)
        skips = []
        for enc in self.encoders:
            feat, skip = enc(feat)
            skips.append(skip)

        g_map = self.global_encoder(global_x)
        g_feat_aligned = spatial_align_crop(g_map, bbox_norm, out_size=feat.shape[-2:])

        c_feat = self.coord_proj(coords).unsqueeze(-1).unsqueeze(-1)
        c_feat_expand = c_feat.expand(-1, -1, feat.shape[2], feat.shape[3])

        fused = torch.cat([feat, g_feat_aligned, c_feat_expand], dim=1)
        fused = self.bottleneck_fuse(fused)
        fused = self.bottleneck(fused, edge_map)

        for dec, skip in zip(self.decoders, reversed(skips)):
            fused = dec(fused, skip)

        logits = self.head(fused)
        return F.interpolate(logits, size=local_x.shape[-2:], mode="bilinear", align_corners=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
