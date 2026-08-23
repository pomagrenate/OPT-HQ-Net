"""
Focal Loss for pixel-wise binary classification.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Focal loss down-weights well-classified easy examples and focuses training
on hard mis-classified pixels — critical when filament pixels represent
< 5% of the image area and background dominates.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

where:
    p_t = p     if y = 1
    p_t = 1 - p if y = 0

Parameters
----------
alpha : float
    Weighting factor for positive class (filament pixels).
    Typical values: 0.25 – 0.75.
gamma : float
    Focusing parameter.  γ = 0 → standard cross-entropy.
    Typical values: 1.5 – 2.5.
reduction : str
    'mean' | 'sum' | 'none'.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Sigmoid focal loss for binary pixel classification.

    Parameters
    ----------
    alpha : float
        Balancing factor for positive (foreground) class.
    gamma : float
        Focusing exponent.  Higher γ → more down-weighting of easy examples.
    reduction : str
        How to reduce the per-pixel losses ('mean', 'sum', 'none').
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    # ------------------------------------------------------------------
    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        logits  : Tensor — raw (un-sigmoidised) predictions, any shape.
        targets : Tensor — binary ground-truth, same shape as logits.
                  Values must be in {0, 1} (float).

        Returns
        -------
        Tensor — scalar (if reduction='mean' or 'sum') or same shape as input.
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
