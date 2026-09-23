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


def soft_skeletonize(img: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
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
    n_iter: int = 5,
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


def tversky_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky index generalizes Dice with independent FP/FN weights.
    Filaments cover a tiny fraction of a tile, so plain Dice lets the
    network drive false negatives to zero without much loss improvement.
    beta > alpha makes false negatives (missed filament pixels) cost more
    than false positives, which is what we want here.
    """
    tp = (pred_prob * target).sum(dim=(1, 2, 3))
    fp = (pred_prob * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_prob) * target).sum(dim=(1, 2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def focal_tversky_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.33,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Focal Tversky (Abraham & Khan, 2019): raises (1 - Tversky) to the
    power 1/gamma so that easy, already-mostly-correct samples (e.g. tiles
    with no filament at all) contribute less, and the network keeps getting
    a meaningful gradient on the hard, rare, thin-filament pixels instead of
    the loss saturating near zero from empty tiles.
    """
    tv = tversky_loss(pred_prob, target, alpha=alpha, beta=beta, smooth=smooth)
    # tv is already averaged over the batch; apply the focal exponent to the
    # per-sample-mean loss (1 - tversky index) rather than post-mean, which
    # keeps this a drop-in, numerically stable replacement for dice_loss.
    return torch.pow(tv.clamp_min(1e-6), 1.0 / gamma)


def masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight_cap: float = 50.0,
) -> torch.Tensor:
    """BCE with a per-batch dynamic pos_weight.

    Filament pixels are typically well under 1% of a tile. Unweighted BCE
    averages over every pixel, so the handful of positive pixels contribute
    a vanishingly small share of the gradient — the network can (and does)
    settle into predicting all-zero, since that already near-minimizes the
    averaged loss. pos_weight upweights the positive-pixel term by roughly
    the negative:positive pixel ratio (capped so a totally empty tile, where
    that ratio is huge, can't blow up the loss/gradients).
    """
    with torch.no_grad():
        pos = (target * valid_mask).sum()
        neg = valid_mask.sum() - pos
        pos_weight = torch.clamp(neg / pos.clamp_min(1.0), max=pos_weight_cap)

    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    loss = loss * valid_mask
    return loss.sum() / valid_mask.sum().clamp_min(1.0)


def cl_dice_weight_schedule(
    epoch: int,
    warmup_epochs: int = 10,
    target_weight: float = 0.5,
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
        w_cldice_target: float = 0.5,
        w_boundary: float = 0.3,
        cldice_warmup_epochs: int = 10,
        skel_iters: int = 5,
        bce_pos_weight_cap: float = 50.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.33,
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_cldice_target = w_cldice_target
        self.w_boundary = w_boundary
        self.cldice_warmup_epochs = cldice_warmup_epochs
        self.skel_iters = skel_iters
        self.bce_pos_weight_cap = bce_pos_weight_cap
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma

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

        l_bce = masked_bce(logits, target, valid_mask, pos_weight_cap=self.bce_pos_weight_cap)
        l_dice = focal_tversky_loss(
            prob_masked,
            target_masked,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.tversky_gamma,
        )
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