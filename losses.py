from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, n_iter: int = 4) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(n_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cl_dice(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    n_iter: int = 4,
    smooth: float = 1.0,
) -> torch.Tensor:
    skel_pred = soft_skeletonize(pred_prob, n_iter)
    skel_true = soft_skeletonize(target, n_iter)

    t_prec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    t_sens = (torch.sum(skel_true * pred_prob) + smooth) / (torch.sum(skel_true) + smooth)

    return 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + 1e-8)


def dice_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    inter = (pred_prob * target).sum(dim=(1, 2, 3))
    union = pred_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight_val: float = 8.0,
) -> torch.Tensor:
    pw = torch.tensor([pos_weight_val], device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pw, reduction="none"
    )
    loss = loss * valid_mask
    return loss.sum() / valid_mask.sum().clamp_min(1.0)


def cl_dice_weight_schedule(
    epoch: int,
    warmup_epochs: int = 6,
    target_weight: float = 0.3,
) -> float:
    if epoch < warmup_epochs:
        return 0.0
    ramp = min(1.0, (epoch - warmup_epochs) / max(1, warmup_epochs))
    return target_weight * ramp


class MicroFilNetLoss(nn.Module):
    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_cldice_target: float = 0.3,
        w_boundary: float = 0.2,
        cldice_warmup_epochs: int = 6,
        skel_iters: int = 4,
        bce_pos_weight: float = 8.0,
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_cldice_target = w_cldice_target
        self.w_boundary = w_boundary
        self.cldice_warmup_epochs = cldice_warmup_epochs
        self.skel_iters = skel_iters
        self.bce_pos_weight = bce_pos_weight

        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def _sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)

    def _boundary_loss(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge_pred = self._sobel_edges(pred_prob)
        edge_true = self._sobel_edges(target)
        return F.l1_loss(edge_pred, edge_true)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = logits.float()
        target = target.float()
        valid_mask = valid_mask.float()

        prob = torch.sigmoid(logits)
        prob_masked = prob * valid_mask
        target_masked = target * valid_mask

        l_bce = masked_bce(logits, target, valid_mask, pos_weight_val=self.bce_pos_weight)
        l_dice = dice_loss(prob_masked, target_masked)
        l_bnd = self._boundary_loss(prob_masked, target_masked)

        w_cl = cl_dice_weight_schedule(epoch, self.cldice_warmup_epochs, self.w_cldice_target)
        if w_cl > 0.0:
            l_cl = soft_cl_dice(prob_masked, target_masked, n_iter=self.skel_iters)
        else:
            l_cl = torch.zeros((), device=logits.device, dtype=torch.float32)

        total = (
            self.w_bce * l_bce
            + self.w_dice * l_dice
            + w_cl * l_cl
            + self.w_boundary * l_bnd
        )

        parts = {
            "bce": l_bce.detach(),
            "dice": l_dice.detach(),
            "cldice": l_cl.detach(),
            "cldice_w": torch.tensor(w_cl, device=logits.device),
            "boundary": l_bnd.detach(),
            "total": total.detach(),
        }
        return total, parts