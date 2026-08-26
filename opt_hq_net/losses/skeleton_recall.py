"""
Skeleton-Recall Loss — topological continuity constraint.

Mathematical formulation
------------------------
Let  y ∈ {0,1}^{H×W}  be the binary ground-truth mask and
     ŷ ∈ [0,1]^{H×W}  the predicted probability map.

The 1D topological skeleton K(y) is extracted by applying the Lee et al.
morphological thinning algorithm (``skimage.morphology.skeletonize``)
to the ground-truth mask.

Skeleton-Recall Loss:

    L_skeleton(y, ŷ) = 1 - Σ_{i ∈ K(y)} y_i · ŷ_i
                            ─────────────────────────
                              Σ_{i ∈ K(y)} y_i  + ε

This quantity penalises the network when it assigns low predicted
probability to skeleton pixels — i.e., when it severs the thin
filament spine by predicting a zero-probability gap.

The loss is computed independently per instance and then averaged.

Reference
---------
Shit et al., "clDice — a Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation", CVPR 2021.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class SkeletonRecallLoss(nn.Module):
    """
    Compute the skeleton-recall loss for binary instance segmentation.

    The skeleton is extracted from the ground-truth mask using morphological
    thinning.  The loss measures how well the predicted probability map
    *covers* the skeleton pixels.

    Parameters
    ----------
    eps : float
        Numerical stability constant added to the denominator.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

        if not SKIMAGE_AVAILABLE:
            raise ImportError(
                "scikit-image is required for SkeletonRecallLoss. "
                "Install with: pip install scikit-image"
            )

    # ------------------------------------------------------------------
    def forward(
        self, pred_logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute skeleton-recall loss.

        Parameters
        ----------
        pred_logits : Tensor (N, 1, H, W) or (N, H, W)
            Raw logits (before sigmoid).
        targets : Tensor (N, H, W) — binary ground-truth masks, values {0, 1}.

        Returns
        -------
        Tensor (scalar) — mean skeleton-recall loss over N instances.
        """
        if pred_logits.dim() == 4:
            pred_logits = pred_logits.squeeze(1)   # (N, H, W)

        pred_logits_clamped = torch.clamp(pred_logits, min=-10.0, max=10.0)
        pred_prob = torch.sigmoid(pred_logits_clamped)      # ŷ ∈ [0,1]

        N = pred_prob.shape[0]
        if N == 0:
            return pred_prob.new_zeros(1).squeeze()

        loss_sum = pred_prob.new_zeros(1)

        for i in range(N):
            gt_mask = targets[i]                    # (H, W) bool/float
            pred_i = pred_prob[i]                   # (H, W) float

            # Compute skeleton K(y) on CPU using scikit-image
            skeleton_mask = self._skeletonize(gt_mask)  # (H, W) bool, on same device

            n_skel = skeleton_mask.float().sum()
            if n_skel < 1:
                # No skeleton pixels — skip (zero loss contribution)
                continue

            # Recall along skeleton pixels
            recall = (skeleton_mask.float() * pred_i).sum() / (n_skel + self.eps)
            recall = torch.clamp(recall, min=0.0, max=1.0)
            loss_sum = loss_sum + (1.0 - recall)

        return loss_sum / max(N, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _skeletonize(mask: torch.Tensor) -> torch.Tensor:
        """
        Apply morphological thinning to a single binary mask tensor.

        Parameters
        ----------
        mask : Tensor (H, W)

        Returns
        -------
        Tensor (H, W) bool, on the same device as input.
        """
        device = mask.device
        # Convert to numpy uint8 for skimage
        mask_np = (mask.detach().cpu().float() > 0.5).numpy().astype(np.uint8)

        # skimage.morphology.skeletonize expects a boolean array
        skel_np = skeletonize(mask_np.astype(bool))

        return torch.from_numpy(skel_np.astype(np.float32)).bool().to(device)
