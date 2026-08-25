"""
Focal-Tversky Loss (FTL) for highly imbalanced binary mask segmentation.

Mathematical Formulation
------------------------
Tversky Index:
    TI = (Σ p_i g_i + ε) / (Σ p_i g_i + α Σ p_i (1 - g_i) + β Σ (1 - p_i) g_i + ε)

Focal-Tversky Loss:
    L_FTL = (1 - TI)^γ

Where:
    - p_i : predicted probability after sigmoid
    - g_i : ground truth binary mask (0 or 1)
    - α   : weight for False Positives (default 0.3)
    - β   : weight for False Negatives (default 0.7 — penalizes missed filaments 2.33× harder)
    - γ   : focal non-linearity parameter (default 0.75)
    - ε   : smooth factor for numerical stability
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky Loss module.

    Parameters
    ----------
    alpha : float
        Weight assigned to False Positives (FP). Default: 0.3.
    beta : float
        Weight assigned to False Negatives (FN). Default: 0.7.
    gamma : float
        Focal exponent to control hard sample weighting. Default: 0.75.
    smooth : float
        Smoothing factor to prevent division by zero. Default: 1e-6.
    from_logits : bool
        If True, applies sigmoid to inputs first. Default: True.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor (N, 1, H, W) or (N, H, W)
            Predicted logits (before or after sigmoid depending on from_logits).
        targets : Tensor (N, 1, H, W) or (N, H, W)
            Ground truth binary masks (0 or 1).

        Returns
        -------
        Tensor (scalar)
            Focal-Tversky loss value.
        """
        if self.from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        # Ensure matching shapes and float dtype
        probs = probs.squeeze(1) if probs.ndim == 4 and probs.shape[1] == 1 else probs
        targets = targets.squeeze(1) if targets.ndim == 4 and targets.shape[1] == 1 else targets
        targets = targets.float()

        # Flatten tensors per batch item or globally
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # True Positives, False Positives, False Negatives
        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = (1.0 - tversky) ** self.gamma

        return tversky_loss
