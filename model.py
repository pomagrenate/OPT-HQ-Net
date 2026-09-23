# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Standard convolution with GroupNorm and SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        c1, c2, k, s, g = int(c1), int(c2), int(k), int(s), int(g)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.norm = nn.GroupNorm(min(8, c2), c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def autopad(k):
    """Pad to 'same' output dimensions when not specified."""
    if isinstance(k, int):
        return k // 2
    if isinstance(k, (list, tuple)):
        return [x // 2 for x in k]
    return int(k) // 2


class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k, 1)
        self.cv2 = Conv(c_, c2, k, 1, g=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """C3k2 module with CSP bottleneck - simplified version."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.cv2 = Conv(c1, self.c_, 1, 1)
        self.cv3 = Conv(2 * self.c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class FilamentSegmentation(nn.Module):
    """
    Solar Filament Segmentation Model based on YOLO26 architecture.
    
    Features:
    - P2/P3/P4/P5 feature pyramid for multi-scale understanding
    - PANet neck for feature fusion
    - Lightweight segmentation head
    - Full 2048x2048 input support
    - Optimized for thin filament detection
    """
    
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Backbone
        self.backbone = self._build_backbone()
        
        # Neck
        self.neck = self._build_neck()
        
        # Segmentation head
        self.head = self._build_head()

    def _build_backbone(self):
        """Build backbone network."""
        return nn.ModuleList([
            Conv(self.in_channels, 32, 3, 2),  # 0: 2048->1024
            Conv(32, 64, 3, 2),                # 1: 1024->512 (P2)
            Conv(64, 128, 3, 2),               # 2: 512->256 (P3)
            Conv(128, 192, 3, 2),              # 3: 256->128 (P4)
            Conv(192, 256, 3, 2),              # 4: 128->64 (P5)
            SPPF(256, 256, 5),                # 5
        ])

    def _build_neck(self):
        """Build PANet neck."""
        return nn.ModuleList([
            nn.Upsample(None, 2, "nearest"),  # 0
            Conv(448, 192, 1, 1, 0),          # 1: reduce concat channels (256+192)
            nn.Upsample(None, 2, "nearest"),  # 2
            Conv(320, 128, 1, 1, 0),          # 3: reduce concat channels (192+128)
            nn.Upsample(None, 2, "nearest"),  # 4
            Conv(192, 64, 1, 1, 0),           # 5: reduce concat channels (128+64)
            Conv(64, 64, 3, 2),                # 6
            Conv(192, 128, 1, 1, 0),          # 7: reduce concat channels (64+128)
            Conv(128, 128, 3, 2),              # 8
            Conv(320, 192, 1, 1, 0),          # 9: reduce concat channels (128+192)
        ])

    def _build_head(self):
        """Build segmentation head."""
        head = nn.ModuleList([
            Conv(64, 16, 1, 1, 0),             # 0: reduce to prototype channels
            Conv(16, 16, 3, 1),                # 1: prototype refinement
            nn.Upsample(None, 2, "bilinear"), # 2: 512->1024
            Conv(16, 16, 3, 1),                # 3: refinement at 1024
            nn.Upsample(None, 2, "bilinear"), # 4: 1024->2048
            nn.Conv2d(16, self.num_classes, 1, 1, 0),  # 5: final mask
        ])
        # Initialize bias for better convergence
        nn.init.constant_(head[5].bias, -2.0)
        return head

    def forward(self, x):
        """Forward pass."""
        # Backbone
        x = self.backbone[0](x)        # 0: 2048->1024
        p2 = self.backbone[1](x)        # 1: 1024->512 (P2)
        p3 = self.backbone[2](p2)       # 2: 512->256 (P3)
        p4 = self.backbone[3](p3)       # 3: 256->128 (P4)
        p5 = self.backbone[4](p4)       # 4: 128->64 (P5)
        p5 = self.backbone[5](p5)       # 5: SPPF

        # Neck - Top-down
        x = self.neck[0](p5)            # 0: upsample
        x = torch.cat([x, p4], dim=1)   # concat P4 (256+192=448)
        p4_fused = self.neck[1](x)      # 1: reduce channels

        x = self.neck[2](p4_fused)      # 2: upsample
        x = torch.cat([x, p3], dim=1)   # concat P3 (192+128=320)
        p3_fused = self.neck[3](x)      # 3: reduce channels

        x = self.neck[4](p3_fused)      # 4: upsample
        x = torch.cat([x, p2], dim=1)   # concat P2 (128+64=192)
        p2_fused = self.neck[5](x)      # 5: reduce channels

        # Neck - Bottom-up
        x = self.neck[6](p2_fused)      # 6: downsample
        x = torch.cat([x, p3_fused], dim=1)  # concat P3 (64+128=192)
        p3_final = self.neck[7](x)      # 7: reduce channels

        x = self.neck[8](p3_final)      # 8: downsample
        x = torch.cat([x, p4_fused], dim=1)  # concat P4 (128+192=320)
        p4_final = self.neck[9](x)      # 9: reduce channels

        # Head
        x = self.head[0](p2_fused)      # 0: reduce to 16 channels
        x = self.head[1](x)             # 1: prototype refinement
        x = self.head[2](x)             # 2: upsample to 1024
        x = self.head[3](x)             # 3: refinement
        x = self.head[4](x)             # 4: upsample to 2048
        mask = self.head[5](x)          # 5: final mask

        return mask


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = FilamentSegmentation(in_channels=1, num_classes=1)
    print(f"Total Parameters: {count_parameters(model):,}")
    
    # Test with single channel input
    dummy_input = torch.randn(1, 1, 2048, 2048)
    mask = model(dummy_input)
    print(f"Mask Logits Shape: {tuple(mask.shape)}")
    
    # Test with two channel input (raw + enhanced)
    model_2ch = FilamentSegmentation(in_channels=2, num_classes=1)
    dummy_2ch = torch.randn(1, 2, 2048, 2048)
    mask_2ch = model_2ch(dummy_2ch)
    print(f"Two-channel Mask Logits Shape: {tuple(mask_2ch.shape)}")