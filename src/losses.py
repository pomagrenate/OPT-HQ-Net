"""
High-Performance, Vectorized Loss Functions for Solar Filament Micro-Segmentation.

Includes:
  - SoftDiceLoss: Memory-efficient soft Dice loss with smooth gradients.
  - SoftclDiceLoss: 100% GPU-differentiable soft centerline Dice loss (Shit et al., CVPR 2021)
                    using morphological min/max pooling (zero CPU skeletonization overhead).
  - CompoundLoss: Multi-task objective combining pixel-level BCE, region-level Dice,
                  topological clDice, and auxiliary skeleton supervision.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GPU Differentiable Soft-Skeletonization via Morphological Pooling
# ---------------------------------------------------------------------------

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological min-pooling (erosion) on GPU.
    Uses directional 1D max-pools to approximate 3x3 structuring element.
    """
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological max-pooling (dilation) on GPU.
    """
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable morphological opening: dilation(erosion(img)).
    """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """
    Extracts continuous, differentiable 1D topological centerlines on GPU.

    Parameters
    ----------
    img : torch.Tensor
        Sigmoid probabilities in range [0, 1], shape (B, 1, H, W).
    iters : int
        Number of morphological thinning iterations (default: 5).

    Returns
    -------
    torch.Tensor
        Continuous skeleton / centerline map in range [0, 1], shape (B, 1, H, W).
    """
    img = torch.clamp(img, 0.0, 1.0)
    opened = soft_open(img)
    skel = F.relu(img - opened)
    current = img

    for _ in range(iters):
        current = soft_erode(current)
        opened_curr = soft_open(current)
        skel = skel + F.relu(current - opened_curr)

    return torch.clamp(skel, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Loss Components
# ---------------------------------------------------------------------------

class SoftDiceLoss(nn.Module):
    """
    Numerically stable Soft Dice Loss with Laplace smoothing.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        probs : Tensor (B, 1, H, W) in [0, 1]
        targets : Tensor (B, 1, H, W) in {0, 1}
        """
        p_flat = probs.contiguous().view(probs.shape[0], -1)
        t_flat = targets.contiguous().view(targets.shape[0], -1)

        intersection = (p_flat * t_flat).sum(dim=1)
        cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class SoftclDiceLoss(nn.Module):
    """
    Differentiable Centerline Dice (clDice) Loss computed entirely on GPU.
    Preserves topological continuity and penalizes filament fragmentation.
    """

    def __init__(self, iters: int = 5, eps: float = 1e-5) -> None:
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        probs : Tensor (B, 1, H, W) in [0, 1]
        targets : Tensor (B, 1, H, W) in {0, 1}
        """
        # Soft-skeletonize prediction and ground-truth on GPU
        skel_pred = soft_skeletonize(probs, iters=self.iters)
        skel_true = soft_skeletonize(targets, iters=self.iters)

        # Topological Precision: fraction of predicted skeleton falling on true mask
        tprec = (skel_pred * targets).sum(dim=(-2, -1)) / (skel_pred.sum(dim=(-2, -1)) + self.eps)

        # Topological Sensitivity: fraction of true skeleton covered by predicted mask
        tsens = (probs * skel_true).sum(dim=(-2, -1)) / (skel_true.sum(dim=(-2, -1)) + self.eps)

        cl_dice = (2.0 * tprec * tsens + self.eps) / (tprec + tsens + self.eps)
        return 1.0 - cl_dice.mean()


class CompoundLoss(nn.Module):
    """
    Unified Multi-Task Loss for High-Precision Solar Filament Segmentation.

    Combines:
      - w_bce   * BCE(mask_logits, targets)
      - w_dice  * SoftDice(mask_probs, targets)
      - w_cldice* SoftclDice(mask_probs, targets)
      - w_skel  * BCE(skel_logits, soft_skeleton(targets))
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_cldice: float = 0.5,
        w_skel: float = 0.5,
        cldice_iters: int = 3,  # Reduced from 5 to 3 for speed
    ) -> None:
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_cldice = w_cldice
        self.w_skel = w_skel
        self.cldice_iters = cldice_iters

        self.dice_loss = SoftDiceLoss()
        self.cldice_loss = SoftclDiceLoss(iters=cldice_iters)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        preds : Tensor (B, 2, H, W)
            Channel 0: mask logits, Channel 1: skeleton logits.
        targets : Tensor (B, 1, H, W) or (B, H, W)
            Ground-truth binary mask (0 or 1).
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Clamp logits for FP16 numerical stability
        mask_logits = torch.clamp(preds[:, 0:1], min=-10.0, max=10.0)
        skel_logits = torch.clamp(preds[:, 1:2], min=-10.0, max=10.0)

        mask_probs = torch.sigmoid(mask_logits)

        # 1. Pixel-wise BCE
        loss_bce = F.binary_cross_entropy_with_logits(mask_logits, targets)

        # 2. Region-wise Soft-Dice
        loss_dice = self.dice_loss(mask_probs, targets)

        # 3. Topological Centerline clDice
        loss_cldice = self.cldice_loss(mask_probs, targets)

        # 4. Auxiliary Skeleton Supervision (GT skeleton generated on GPU)
        with torch.no_grad():
            target_skel = soft_skeletonize(targets, iters=self.cldice_iters)
        loss_skel = F.binary_cross_entropy_with_logits(skel_logits, target_skel)

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.w_bce * loss_bce
            + self.w_dice * loss_dice
            + self.w_cldice * loss_cldice
            + self.w_skel * loss_skel
        )

        return {
            "loss": total_loss,
            "loss_bce": loss_bce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_cldice": loss_cldice.detach(),
            "loss_skel": loss_skel.detach(),
        }
