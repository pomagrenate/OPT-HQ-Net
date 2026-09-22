"""
Topology- and barb-centric loss for MicroFilNet.

  L_total(epoch) = w_bce * BCE + w_dice * Dice
                 + w_cl(epoch) * soft_clDice
                 + w_edge * BoundaryLoss

Why not just BCE+Dice (what prior filament-segmentation papers use):
Two predictions can achieve identical Dice while one is a single connected
filament and the other is fragmented into five islands -- Dice/BCE are
purely pixel-overlap measures and are blind to topology. soft-clDice
(Shit et al., CVPR 2021) fixes this: it is computed on the *skeletons* of
prediction and ground truth, and is provably sensitive to connectivity
breaks and merges. We warm it in gradually (see `cl_dice_weight_schedule`)
because a raw clDice gradient on a near-random initial mask is unstable --
skeletonization of a noisy mask is itself noisy.

BoundaryLoss re-uses the model's own zero-parameter ridge/edge prior
target (a Sobel/Canny edge map of the ground-truth mask) as extra
pixel-wise supervision on a thin band around filament boundaries, which
sharpens barbs specifically without needing a dedicated barb detector.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Differentiable soft skeletonization (Shit et al., CVPR 2021, official
# morphological formulation, reproduced here for 2D binary/probability maps)
# --------------------------------------------------------------------------

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(n_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cl_dice(pred_prob: torch.Tensor, target: torch.Tensor,
                  n_iter: int = 10, smooth: float = 1.0) -> torch.Tensor:
    """
    pred_prob, target: (B, 1, H, W) in [0,1]. Returns scalar loss in [0,1],
    0 = perfect skeleton/topology match.
    """
    skel_pred = soft_skeletonize(pred_prob, n_iter)
    skel_true = soft_skeletonize(target, n_iter)

    t_prec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    t_sens = (torch.sum(skel_true * pred_prob) + smooth) / (torch.sum(skel_true) + smooth)

    cl_dice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + 1e-8)
    return cl_dice


# --------------------------------------------------------------------------
# Standard region losses
# --------------------------------------------------------------------------

def dice_loss(pred_prob: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    inter = (pred_prob * target).sum(dim=(1, 2, 3))
    union = pred_prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def masked_bce(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * valid_mask
    return loss.sum() / valid_mask.sum().clamp_min(1.0)


# --------------------------------------------------------------------------
# Boundary / barb-sharpening loss
# --------------------------------------------------------------------------

def sobel_edges(mask: torch.Tensor) -> torch.Tensor:
    """Cheap, fixed (non-learned) Sobel edge magnitude of a binary/prob mask."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(mask, kx, padding=1)
    gy = F.conv2d(mask, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def boundary_loss(pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    edge_pred = sobel_edges(pred_prob)
    edge_true = sobel_edges(target)
    return F.l1_loss(edge_pred, edge_true)


# --------------------------------------------------------------------------
# Combined loss with curriculum warmup for the topology term
# --------------------------------------------------------------------------

def cl_dice_weight_schedule(epoch: int, warmup_epochs: int = 10, target_weight: float = 0.5) -> float:
    """
    Ramp the clDice weight in linearly after `warmup_epochs`, once the base
    mask is roughly correct -- skeletonizing a near-random early mask
    produces a noisy, unhelpful gradient.
    """
    if epoch < warmup_epochs:
        return 0.0
    ramp = min(1.0, (epoch - warmup_epochs) / max(1, warmup_epochs))
    return target_weight * ramp


class MicroFilNetLoss(nn.Module):
    def __init__(self, w_bce: float = 1.0, w_dice: float = 1.0,
                 w_cldice_target: float = 0.5, w_boundary: float = 0.3,
                 cldice_warmup_epochs: int = 10, skel_iters: int = 10):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_cldice_target = w_cldice_target
        self.w_boundary = w_boundary
        self.cldice_warmup_epochs = cldice_warmup_epochs
        self.skel_iters = skel_iters

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                valid_mask: torch.Tensor, epoch: int):
        prob = torch.sigmoid(logits)
        prob_masked = prob * valid_mask
        target_masked = target * valid_mask

        l_bce = masked_bce(logits, target, valid_mask)
        l_dice = dice_loss(prob_masked, target_masked)
        l_bnd = boundary_loss(prob_masked, target_masked)

        w_cl = cl_dice_weight_schedule(epoch, self.cldice_warmup_epochs, self.w_cldice_target)
        l_cl = soft_cl_dice(prob_masked, target_masked, n_iter=self.skel_iters) if w_cl > 0 else torch.zeros((), device=logits.device)

        total = (self.w_bce * l_bce + self.w_dice * l_dice
                 + w_cl * l_cl + self.w_boundary * l_bnd)

        parts = {
            "bce": l_bce.detach(), "dice": l_dice.detach(),
            "cldice": l_cl.detach(), "cldice_w": torch.tensor(w_cl),
            "boundary": l_bnd.detach(), "total": total.detach(),
        }
        return total, parts


if __name__ == "__main__":
    B, H, W = 2, 128, 128
    logits = torch.randn(B, 1, H, W, requires_grad=True)
    target = (torch.rand(B, 1, H, W) > 0.9).float()
    valid = torch.ones(B, 1, H, W)
    crit = MicroFilNetLoss()
    loss, parts = crit(logits, target, valid, epoch=15)
    loss.backward()
    print({k: float(v) for k, v in parts.items()})
