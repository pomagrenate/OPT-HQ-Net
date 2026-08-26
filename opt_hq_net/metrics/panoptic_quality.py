"""
Panoptic Quality (PQ), Dice score, and IoU metric implementation.

Mathematical definitions
------------------------
Panoptic Quality:

    PQ(Y, Ŷ) = Σ_{(y,ŷ)∈TP} IoU(y,ŷ)
               ─────────────────────────
               |TP| + 0.5|FP| + 0.5|FN|

where TP requires IoU(y, ŷ) > 0.5.

Instance IoU:

    IoU(y, ŷ) = |y ∩ ŷ| / |y ∪ ŷ|

This implementation supports:
  - Batched evaluation (accumulate over multiple images then compute final).
  - Per-image PQ for monitoring during training.
  - Mean Dice and mean IoU score distributions.

Usage
-----
>>> metric = PanopticQualityMetric(iou_threshold=0.5)
>>> metric.update(pred_masks_list, gt_masks_list)   # call per batch
>>> results = metric.compute()
>>> print(results['PQ'], results['mean_dice'], results['mean_iou'])
>>> metric.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PQAccumulator:
    """Accumulates TP/FP/FN counts and IoU sums across images."""
    tp_count: int = 0
    fp_count: int = 0
    fn_count: int = 0
    iou_sum: float = 0.0
    dice_values: List[float] = field(default_factory=list)
    iou_values: List[float] = field(default_factory=list)


class PanopticQualityMetric:
    """
    Streaming Panoptic Quality metric for solar filament segmentation.

    Accumulates statistics over a full dataset via ``update()`` calls,
    then computes the final PQ via ``compute()``.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a predicted mask to be counted as True Positive.
        Default 0.5 as specified by the MAGFiLO benchmark.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._acc = PQAccumulator()

    # ------------------------------------------------------------------
    def update(
        self,
        pred_masks_list: List[np.ndarray],
        gt_masks_list: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Update the accumulator for one batch of images.

        Parameters
        ----------
        pred_masks_list : list[np.ndarray]
            One entry per image.  Each entry is (N_pred, H, W) uint8.
        gt_masks_list : list[np.ndarray]
            One entry per image.  Each entry is (N_gt, H, W) uint8.

        Returns
        -------
        dict[str, float]
            Per-batch PQ, mean Dice, mean IoU (for logging).
        """
        batch_acc = PQAccumulator()

        for pred_masks, gt_masks in zip(pred_masks_list, gt_masks_list):
            self._process_single_image(pred_masks, gt_masks, batch_acc)

        # Merge into running accumulator
        self._acc.tp_count += batch_acc.tp_count
        self._acc.fp_count += batch_acc.fp_count
        self._acc.fn_count += batch_acc.fn_count
        self._acc.iou_sum += batch_acc.iou_sum
        self._acc.dice_values.extend(batch_acc.dice_values)
        self._acc.iou_values.extend(batch_acc.iou_values)

        return self._compute_from(batch_acc)

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        """Compute final metric values over all accumulated images."""
        return self._compute_from(self._acc)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the accumulator for a new evaluation epoch."""
        self._acc = PQAccumulator()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _process_single_image(
        self,
        pred_masks: np.ndarray,
        gt_masks: np.ndarray,
        acc: PQAccumulator,
    ) -> None:
        """
        Match predictions to ground-truth for one image and update ``acc``.
        """
        n_pred = len(pred_masks)
        n_gt = len(gt_masks)

        if n_gt == 0 and n_pred == 0:
            return

        if n_gt == 0:
            acc.fp_count += n_pred
            return

        if n_pred == 0:
            acc.fn_count += n_gt
            return

        # Compute IoU matrix (n_pred × n_gt)
        iou_mat = self._compute_iou_matrix(pred_masks, gt_masks)  # (P, G)

        matched_pred = set()
        matched_gt = set()

        # Greedy matching: sort all (pred, gt) pairs by IoU descending
        pairs = sorted(
            [(i, j, iou_mat[i, j]) for i in range(n_pred) for j in range(n_gt)],
            key=lambda x: x[2],
            reverse=True,
        )

        for pred_i, gt_j, iou_val in pairs:
            if pred_i in matched_pred or gt_j in matched_gt:
                continue
            if iou_val < self.iou_threshold:
                break   # remaining pairs have even lower IoU

            # True Positive
            acc.tp_count += 1
            acc.iou_sum += iou_val
            acc.iou_values.append(iou_val)

            # Dice from masks
            dice = self._compute_dice(pred_masks[pred_i], gt_masks[gt_j])
            acc.dice_values.append(dice)

            matched_pred.add(pred_i)
            matched_gt.add(gt_j)

        # Unmatched predictions → FP
        acc.fp_count += n_pred - len(matched_pred)
        # Unmatched ground-truth → FN
        acc.fn_count += n_gt - len(matched_gt)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_iou_matrix(
        pred_masks: np.ndarray, gt_masks: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise mask IoU.

        Parameters
        ----------
        pred_masks : (P, H, W) uint8
        gt_masks   : (G, H, W) uint8

        Returns
        -------
        (P, G) float32
        """
        P, H, W = pred_masks.shape
        G = gt_masks.shape[0]

        # Spatial shape matching fallback (e.g. 512x512 pred vs 2048x2048 GT)
        if G > 0 and P > 0:
            Hg, Wg = gt_masks.shape[1], gt_masks.shape[2]
            if (H, W) != (Hg, Wg):
                import cv2
                resized_preds = [
                    cv2.resize(pm, (Wg, Hg), interpolation=cv2.INTER_NEAREST)
                    for pm in pred_masks
                ]
                pred_masks = np.stack(resized_preds, axis=0)
                P, H, W = pred_masks.shape

        iou_mat = np.zeros((P, G), dtype=np.float32)

        # Flatten masks
        pred_flat = pred_masks.reshape(P, -1).astype(bool)   # (P, HW)
        gt_flat = gt_masks.reshape(G, -1).astype(bool)       # (G, HW)

        for i in range(P):
            for j in range(G):
                inter = (pred_flat[i] & gt_flat[j]).sum()
                union = (pred_flat[i] | gt_flat[j]).sum()
                iou_mat[i, j] = inter / union if union > 0 else 0.0

        return iou_mat

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice coefficient for a single matched pair."""
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_b = pred.astype(bool)
        gt_b = gt.astype(bool)
        inter = (pred_b & gt_b).sum()
        total = pred_b.sum() + gt_b.sum()
        return float(2 * inter / total) if total > 0 else 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_from(acc: PQAccumulator) -> Dict[str, float]:
        """Compute PQ, Dice, IoU from a ``PQAccumulator``."""
        denom = acc.tp_count + 0.5 * acc.fp_count + 0.5 * acc.fn_count
        pq = (acc.iou_sum / denom) if denom > 0 else 0.0
        mean_dice = float(np.mean(acc.dice_values)) if acc.dice_values else 0.0
        mean_iou = float(np.mean(acc.iou_values)) if acc.iou_values else 0.0

        return {
            "PQ": round(pq, 4),
            "mean_dice": round(mean_dice, 4),
            "mean_iou": round(mean_iou, 4),
            "TP": acc.tp_count,
            "FP": acc.fp_count,
            "FN": acc.fn_count,
        }
