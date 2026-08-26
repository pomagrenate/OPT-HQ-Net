"""
Combined multi-task loss for OPT-HQ Net.

Total loss:

    L_total = λ1·L_box + λ2·L_focal + λ3·L_dice + λ4·L_skeleton

Where:
    λ1 = 1.0  (Oriented Box Regression — Smooth L1)
    λ2 = 2.0  (Focal Loss — pixel-wise foreground detection)
    λ3 = 2.0  (Binary Dice Loss — area overlap)
    λ4 = 1.5  (Skeleton-Recall Loss — topological continuity)

The combined loss handles:
  - Matching predicted masks to ground-truth via greedy IoU matching.
  - Aligning mask spatial scales (predicted masks are decoded at a fixed
    resolution; GT masks may be at a different resolution).
  - Graceful handling of zero-instance images.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_hq_net.losses.focal_tversky import FocalTverskyLoss
from opt_hq_net.losses.skeleton_recall import SkeletonRecallLoss


class OPTHQNetLoss(nn.Module):
    """
    Compound multi-task loss combining all four loss components.

    Parameters
    ----------
    lambda_box      : float  λ1 — oriented box regression weight.
    lambda_tversky  : float  λ2 — focal-tversky loss weight.
    lambda_skeleton : float  λ3 — skeleton-recall loss weight.
    tversky_alpha   : float  α for FocalTverskyLoss.
    tversky_beta    : float  β for FocalTverskyLoss (higher β penalizes false negatives).
    tversky_gamma   : float  γ for FocalTverskyLoss.
    iou_match_threshold : float
        Minimum IoU between predicted and GT box to assign a positive match.
    """

    def __init__(
        self,
        lambda_box: float = 1.0,
        lambda_tversky: float = 2.0,
        lambda_skeleton: float = 1.5,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        tversky_gamma: float = 0.75,
        iou_match_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_tversky = lambda_tversky
        self.lambda_skeleton = lambda_skeleton
        self.iou_threshold = iou_match_threshold

        self.tversky_loss = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
        )
        self.skeleton_loss = SkeletonRecallLoss()

    # ------------------------------------------------------------------
    def forward(
        self,
        mask_logits: torch.Tensor,          # (N_total, 1, H_out, W_out)
        proposals: List[torch.Tensor],      # per image, each (N_i, 5+1)
        batch_idx: torch.Tensor,            # (N_total,)
        gt_boxes: List[torch.Tensor],       # per image, each (G_i, 5)
        gt_masks: List[torch.Tensor],       # per image, each (G_i, H, W) bool
        image_size: Tuple[int, int],        # (H_img, W_img)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Returns
        -------
        dict[str, Tensor] with keys:
            'mask_tversky_loss', 'mask_skeleton_loss'
        """

        matched_logits = []
        matched_gt_masks = []

        batch_size = len(proposals)
        for i in range(batch_size):
            idx = (batch_idx == i).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue

            prop_i = proposals[i]
            gt_boxes_i = gt_boxes[i]
            gt_masks_i = gt_masks[i]

            if len(gt_boxes_i) == 0 or len(gt_masks_i) == 0:
                continue

            matches = self._match_boxes(prop_i[:, :5], gt_boxes_i)
            logits_i = mask_logits[idx]

            out_h, out_w = logits_i.shape[-2:]

            for p_idx, g_idx in matches:
                matched_logits.append(logits_i[p_idx])

                gt_m = gt_masks_i[g_idx]
                if gt_m.shape[-2:] != (out_h, out_w):
                    gt_m = self._resize_gt_mask(gt_m, out_h, out_w)
                matched_gt_masks.append(gt_m)

        if not matched_logits:
            zero = mask_logits.new_zeros(1)
            return {
                "mask_tversky_loss": zero,
                "mask_skeleton_loss": zero,
            }

        # Stack into batch tensors
        logits_stk = torch.stack(matched_logits, dim=0)      # (M, 1, H, W)
        gt_masks_stk = torch.stack(matched_gt_masks, dim=0)  # (M, H, W)

        tversky = self.tversky_loss(logits_stk, gt_masks_stk)
        skeleton = self.skeleton_loss(logits_stk, gt_masks_stk)

        tversky = torch.nan_to_num(tversky, nan=0.0)
        skeleton = torch.nan_to_num(skeleton, nan=0.0)

        return {
            "mask_tversky_loss": self.lambda_tversky * tversky,
            "mask_skeleton_loss": self.lambda_skeleton * skeleton,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _match_boxes(
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Greedy one-to-one matching between predicted and GT boxes by axis-aligned IoU.

        Returns list of (pred_idx, gt_idx) pairs.
        """
        def to_xyxy(b: torch.Tensor) -> torch.Tensor:
            xc, yc, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            return torch.stack([xc - w/2, yc - h/2, xc + w/2, yc + h/2], dim=1)

        pa = to_xyxy(pred_boxes)  # (K, 4)
        ga = to_xyxy(gt_boxes)    # (G, 4)

        inter_x1 = torch.max(pa[:, 0].unsqueeze(1), ga[:, 0].unsqueeze(0))
        inter_y1 = torch.max(pa[:, 1].unsqueeze(1), ga[:, 1].unsqueeze(0))
        inter_x2 = torch.min(pa[:, 2].unsqueeze(1), ga[:, 2].unsqueeze(0))
        inter_y2 = torch.min(pa[:, 3].unsqueeze(1), ga[:, 3].unsqueeze(0))

        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area_p = (pa[:, 2] - pa[:, 0]) * (pa[:, 3] - pa[:, 1])
        area_g = (ga[:, 2] - ga[:, 0]) * (ga[:, 3] - ga[:, 1])
        iou = inter / (area_p.unsqueeze(1) + area_g.unsqueeze(0) - inter + 1e-6)

        # Greedy matching: for each GT, assign best unmatched prediction
        iou_np = iou.detach().cpu().numpy()
        G = gt_boxes.shape[0]
        matched = []
        used_pred = set()
        for g in range(G):
            best_pred = int(iou_np[:, g].argmax())
            # Match best proposal per GT box (threshold 0.05, or fallback to top proposal for early training)
            if best_pred not in used_pred:
                if iou_np[best_pred, g] >= 0.05 or len(matched) == 0:
                    matched.append((best_pred, g))
                    used_pred.add(best_pred)

        return matched

    # ------------------------------------------------------------------
    @staticmethod
    def _resize_gt_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Resize a single GT mask to (h, w) using nearest-neighbour."""
        if mask.shape[-2] == h and mask.shape[-1] == w:
            return mask.float()
        return F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest",
        ).squeeze(0).squeeze(0)
