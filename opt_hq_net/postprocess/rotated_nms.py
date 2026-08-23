"""
Polygon-based Rotated Non-Maximum Suppression (Rotated NMS).

Standard axis-aligned NMS fails to correctly suppress duplicate proposals
for arbitrarily oriented filaments because two oriented boxes may have
low rectangular IoU yet high true polygon IoU (or vice versa).

This module computes exact rotated polygon intersection-over-union between
oriented bounding boxes using the Shapely library, then applies greedy NMS.

Usage
-----
>>> nms = RotatedNMS(iou_threshold=0.40)
>>> keep = nms(boxes, scores)  # boxes: (N, 5), scores: (N,)
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def oriented_box_to_polygon(box: np.ndarray) -> "Polygon":
    """
    Convert a 5-parameter oriented box to a Shapely Polygon.

    Parameters
    ----------
    box : np.ndarray (5,) — [xc, yc, w, h, θ_rad]

    Returns
    -------
    Shapely Polygon representing the rotated rectangle.
    """
    xc, yc, w, h, theta = box
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Half-dimensions
    hw, hh = w / 2.0, h / 2.0

    # Four corners in local (unrotated) frame
    corners_local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])

    # Rotation matrix
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = corners_local @ R.T + np.array([xc, yc])

    return Polygon(corners_world.tolist())


def rotated_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise polygon IoU matrix for N oriented boxes.

    Parameters
    ----------
    boxes : np.ndarray (N, 5) — [xc, yc, w, h, θ_rad]

    Returns
    -------
    np.ndarray (N, N) — IoU values ∈ [0, 1]
    """
    N = len(boxes)
    polys = [oriented_box_to_polygon(b) for b in boxes]
    iou_mat = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            inter = polys[i].intersection(polys[j]).area
            union = polys[i].area + polys[j].area - inter
            iou = inter / union if union > 0 else 0.0
            iou_mat[i, j] = iou_mat[j, i] = iou

    return iou_mat


class RotatedNMS:
    """
    Greedy Rotated Non-Maximum Suppression using exact polygon IoU.

    Parameters
    ----------
    iou_threshold : float
        Suppress a box if its polygon IoU with a higher-scoring box
        exceeds this threshold.  Recommended: 0.40 for filaments.
    score_threshold : float
        Minimum score to consider a box before NMS.
    max_dets : int
        Maximum number of detections to return after NMS.

    Raises
    ------
    ImportError
        If ``shapely`` is not installed.
    """

    def __init__(
        self,
        iou_threshold: float = 0.40,
        score_threshold: float = 0.05,
        max_dets: int = 300,
    ) -> None:
        if not SHAPELY_AVAILABLE:
            raise ImportError(
                "shapely is required for RotatedNMS. "
                "Install with: pip install shapely"
            )
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_dets = max_dets

    # ------------------------------------------------------------------
    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Rotated NMS.

        Parameters
        ----------
        boxes  : Tensor (N, 5) — [xc, yc, w, h, θ_rad]
        scores : Tensor (N,)

        Returns
        -------
        keep : Tensor (K,) — indices of surviving boxes (LongTensor).
        """
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long)

        # Score threshold filter
        score_keep = scores >= self.score_threshold
        if not score_keep.any():
            return torch.zeros(0, dtype=torch.long)

        boxes_np = boxes[score_keep].detach().cpu().numpy()
        scores_np = scores[score_keep].detach().cpu().numpy()
        original_idx = torch.where(score_keep)[0]

        # Sort by descending score
        order = np.argsort(-scores_np)
        boxes_sorted = boxes_np[order]

        # Build pairwise IoU matrix
        iou_mat = rotated_iou_matrix(boxes_sorted)

        # Greedy suppression
        suppressed = np.zeros(len(order), dtype=bool)
        keep_local: List[int] = []

        for i in range(len(order)):
            if suppressed[i]:
                continue
            keep_local.append(i)
            if len(keep_local) >= self.max_dets:
                break
            # Suppress lower-scoring boxes with high IoU
            suppressed[i + 1:] |= iou_mat[i, i + 1:] >= self.iou_threshold

        # Map back to original indices
        keep_sorted_idx = order[keep_local]
        return original_idx[keep_sorted_idx]
