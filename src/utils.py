"""
Utilities for Metric Evaluation, Kaggle Submission RLE Encoding, and Model EMA.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from pycocotools import mask as mask_utils
    PYCOCO_AVAILABLE = True
except ImportError:
    PYCOCO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kaggle RLE Mask Encoding
# ---------------------------------------------------------------------------

def binary_mask_to_rle(binary_mask: np.ndarray) -> str:
    """
    Encode 2D binary uint8 mask into Kaggle submission RLE string.
    Uses pycocotools Fortran-order encoding with zero-overhead fallback.
    """
    if PYCOCO_AVAILABLE:
        fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        rle_dict = mask_utils.encode(fortran_mask)
        return rle_dict["counts"].decode("utf-8")

    # Vectorized pure Python/NumPy RLE fallback (matches pycocotools Fortran column-major order)
    dots = binary_mask.astype(bool).flatten(order="F")
    runs = np.where(dots[1:] != dots[:-1])[0] + 2
    prefix = [1] if dots[0] else []
    suffix = [len(dots) + 1] if dots[-1] else []
    runs = np.concatenate([prefix, runs, suffix]).astype(np.int64)
    lengths = runs[1::2] - runs[::2]
    starts = runs[::2]
    return " ".join(f"{int(s)} {int(l)}" for s, l in zip(starts, lengths))


def rle_to_binary_mask(rle_str: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode RLE string into binary uint8 mask (H, W)."""
    if not rle_str or not str(rle_str).strip():
        return np.zeros((height, width), dtype=np.uint8)

    if PYCOCO_AVAILABLE:
        try:
            rle_dict = {"size": [height, width], "counts": str(rle_str).encode("utf-8")}
            return mask_utils.decode(rle_dict)
        except Exception:
            pass

    s = str(rle_str).split()
    starts = [int(float(x)) for x in s[0:][::2]]
    lengths = [int(float(x)) for x in s[1:][::2]]
    img = np.zeros(height * width, dtype=np.uint8)
    for st, le in zip(starts, lengths):
        st_0 = st - 1
        img[st_0 : st_0 + le] = 1
    return img.reshape((height, width), order="F")


# ---------------------------------------------------------------------------
# Evaluation Metrics: Mean Dice & Panoptic Quality (PQ)
# ---------------------------------------------------------------------------

def compute_mean_dice(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-5) -> float:
    """Compute mean Dice score across batch."""
    p = (preds > 0.5).astype(np.float32)
    t = (targets > 0.5).astype(np.float32)

    inter = (p * t).sum()
    card = p.sum() + t.sum()
    if card == 0:
        return 1.0
    return float((2.0 * inter + eps) / (card + eps))


class PanopticQualityMetric:
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
    for individual filament instances using IoU > 0.5 bipartite matching.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        self.total_iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.dice_scores: List[float] = []

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> None:
        """
        Parameters
        ----------
        pred_mask : np.ndarray (H, W) binary {0, 1}
        gt_mask   : np.ndarray (H, W) binary {0, 1}
        """
        # Pixel Dice
        self.dice_scores.append(compute_mean_dice(pred_mask, gt_mask))

        # Extract instances via connected components
        _, pred_labels = cv2.connectedComponents((pred_mask > 0).astype(np.uint8))
        _, gt_labels = cv2.connectedComponents((gt_mask > 0).astype(np.uint8))

        pred_insts = [(pred_labels == k).astype(np.uint8) for k in range(1, pred_labels.max() + 1)]
        gt_insts = [(gt_labels == k).astype(np.uint8) for k in range(1, gt_labels.max() + 1)]

        matched_gt = set()
        matched_pred = set()

        for p_idx, p_arr in enumerate(pred_insts):
            best_iou = 0.0
            best_g_idx = -1
            p_area = p_arr.sum()

            for g_idx, g_arr in enumerate(gt_insts):
                if g_idx in matched_gt:
                    continue
                intersection = (p_arr & g_arr).sum()
                union = p_area + g_arr.sum() - intersection
                iou = intersection / max(union, 1)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou > self.iou_threshold and best_g_idx != -1:
                self.tp += 1
                self.total_iou += best_iou
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)

        self.fp += len(pred_insts) - len(matched_pred)
        self.fn += len(gt_insts) - len(matched_gt)

    def compute(self) -> Dict[str, float]:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        pq = (self.total_iou / denom) if denom > 0 else 0.0
        sq = (self.total_iou / max(self.tp, 1)) if self.tp > 0 else 0.0
        rq = (self.tp / denom) if denom > 0 else 0.0
        mean_dice = float(np.mean(self.dice_scores)) if self.dice_scores else 0.0

        return {
            "PQ": pq,
            "SQ": sq,
            "RQ": rq,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "mean_dice": mean_dice,
        }


# ---------------------------------------------------------------------------
# Model EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Exponential Moving Average of model parameters to stabilize validation metrics.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cuda") -> None:
        self.module = copy.deepcopy(model).eval().to(device)
        self.decay = decay
        self.device = device
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(self.decay * v + (1.0 - self.decay) * msd[k].to(self.device))

    @torch.no_grad()
    def set(self, model: nn.Module) -> None:
        self.module.load_state_dict(model.state_dict())
