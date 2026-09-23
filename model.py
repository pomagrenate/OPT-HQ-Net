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
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class StripPooling(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv_h = nn.Conv2d(ch, ch, 1, bias=False)
        self.conv_v = nn.Conv2d(ch, ch, 1, bias=False)
        self.fuse = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_h = self.conv_h(F.adaptive_avg_pool2d(x, (h, 1)))
        x_v = self.conv_v(F.adaptive_avg_pool2d(x, (1, w)))
        strip = self.fuse(x_h + x_v)
        gate = torch.sigmoid(self.bn(strip))
        return x + x * gate


class MultiScaleDilationMixer(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        sub_ch = ch // 4
        self.sub_ch = sub_ch
        self.d1 = DWSepConv(sub_ch, sub_ch, dilation=1)
        self.d2 = DWSepConv(sub_ch, sub_ch, dilation=2)
        self.d4 = DWSepConv(sub_ch, sub_ch, dilation=4)
        self.d8 = DWSepConv(sub_ch, sub_ch, dilation=8)
        self.strip = StripPooling(ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, self.sub_ch, dim=1)
        o1 = self.d1(xs[0])
        o2 = self.d2(xs[1])
        o4 = self.d4(xs[2])
        o8 = self.d8(xs[3])
        out = torch.cat([o1, o2, o4, o8], dim=1)
        return self.strip(self.fuse(out))


class HighResDetailStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.SiLU(inplace=True),
            DWSepConv(4, 4),
        )
        self.down = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            DWSepConv(8, 8),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_2048 = self.stem(x)
        f_1024 = self.down(f_2048)
        return f_2048, f_1024


class CoarseContextStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            DWSepConv(16, 24),
            nn.MaxPool2d(2),
            DWSepConv(24, 32),
            nn.MaxPool2d(2),
            DWSepConv(32, 48),
        )
        self.down_stage = nn.Sequential(
            nn.MaxPool2d(2),
            DWSepConv(48, 64),
            nn.MaxPool2d(2),
            DWSepConv(64, 96),
        )
        self.mixer = MultiScaleDilationMixer(96)

    def forward(self, x_256: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_32 = self.stem(x_256)
        f_8 = self.down_stage(f_32)
        f_8 = self.mixer(f_8)
        return f_32, f_8


class MicroFilNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_res_stream = HighResDetailStream()
        self.context_stream = CoarseContextStream()

        self.global_gate_1024 = nn.Sequential(
            nn.Conv2d(96, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.Sigmoid(),
        )

        self.global_gate_2048 = nn.Sequential(
            nn.Conv2d(48, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
        )

        self.refine_1024 = nn.Sequential(
            DWSepConv(8, 8),
            DWSepConv(8, 8),
        )

        self.up_to_2048 = nn.Sequential(
            nn.Conv2d(8 + 4, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            DWSepConv(8, 4),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.SiLU(inplace=True),
            nn.Conv2d(4, 1, 1),
        )

        self.boundary_head = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.SiLU(inplace=True),
            nn.Conv2d(4, 1, 1),
        )

        nn.init.constant_(self.mask_head[-1].bias, -2.0)
        nn.init.constant_(self.boundary_head[-1].bias, -2.0)

    def forward(self, x_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_coarse = F.interpolate(x_full, size=(256, 256), mode="bilinear", align_corners=False)

        f_2048, f_1024 = self.high_res_stream(x_full)
        f_coarse_32, f_coarse_8 = self.context_stream(x_coarse)

        gate_1024 = F.interpolate(f_coarse_8, size=f_1024.shape[-2:], mode="bilinear", align_corners=False)
        gate_1024 = self.global_gate_1024(gate_1024)
        f_1024 = self.refine_1024(f_1024 * gate_1024 + f_1024)

        f_1024_up = F.interpolate(f_1024, size=f_2048.shape[-2:], mode="bilinear", align_corners=False)
        gate_2048 = F.interpolate(f_coarse_32, size=f_2048.shape[-2:], mode="bilinear", align_corners=False)
        gate_2048 = self.global_gate_2048(gate_2048)
        f_2048 = f_2048 * gate_2048 + f_2048

        f_final = self.up_to_2048(torch.cat([f_1024_up, f_2048], dim=1))

        mask_logits = self.mask_head(f_final)
        boundary_logits = self.boundary_head(f_final)

        return mask_logits, boundary_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MicroFilNet()
    print(f"Total Parameters: {count_parameters(net):,}")
    
    dummy_input = torch.randn(1, 1, 2048, 2048)
    mask, boundary = net(dummy_input)
    print(f"Mask Logits Shape: {tuple(mask.shape)}")
    print(f"Boundary Logits Shape: {tuple(boundary.shape)}")