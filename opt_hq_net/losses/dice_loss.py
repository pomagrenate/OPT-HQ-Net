"""
Binary Dice Loss for instance mask segmentation.

Dice loss maximises the Sørensen–Dice coefficient between predicted
probability maps and binary ground-truth masks.  It is naturally
class-balanced — the numerator and denominator both scale with the
foreground area, so small filament instances are weighted fairly.

    L_dice(y, ŷ) = 1 - 2 · Σ(y · ŷ) / (Σy + Σŷ + ε)

This implementation:
  - Operates on raw logits (sigmoid applied internally).
  - Supports per-instance computation followed by mean aggregation.
  - Handles empty masks (N=0) gracefully.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice loss for pixel-wise binary segmentation.

    Parameters
    ----------
    eps : float
        Numerical stability constant for the denominator.
    reduction : str
        'mean' — average Dice loss across instances.
        'sum'  — total Dice loss.
        'none' — per-instance Dice loss tensor.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    # ------------------------------------------------------------------
    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary Dice loss.

        Parameters
        ----------
        logits  : Tensor (N, 1, H, W) or (N, H, W) — raw predictions.
        targets : Tensor (N, H, W) — binary ground-truth, values {0, 1}.

        Returns
        -------
        Tensor — scalar or (N,) depending on reduction.
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)     # (N, H, W)

        probs = torch.sigmoid(logits)       # ŷ ∈ [0, 1]
        targets = targets.float()

        # Flatten spatial dimensions
        N = probs.shape[0]
        probs_flat = probs.reshape(N, -1)       # (N, H*W)
        targets_flat = targets.reshape(N, -1)   # (N, H*W)

        intersection = (probs_flat * targets_flat).sum(dim=1)  # (N,)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)  # (N,)

        dice_score = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice_score   # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
