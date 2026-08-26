"""
Instance Grouping & Post-Processing Module for Filament-HQ.

Resolves over-segmentation and false-positive fragmentation:
  1. Hysteresis Thresholding (high-confidence seeds + candidate propagation)
  2. Minimum Area Filtering (discards tiny noise fragments < 50 pixels)
  3. Boundary-Aware Seed Isolation (prevents false over-merging while preserving long filaments)
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class FilamentPostProcessor:
    """
    Filament Instance Grouping Post-Processor.

    Parameters
    ----------
    seed_thresh : float
        Semantic threshold for core instance seeds (default 0.65).
    mask_thresh : float
        Semantic threshold for candidate filament pixels (default 0.35).
    min_area : int
        Minimum pixel area to qualify as a valid filament instance (default 50).
    bnd_suppress_thresh : float
        Boundary threshold to prevent instance leakage across clear borders (default 0.5).
    """

    def __init__(
        self,
        seed_thresh: float = 0.65,
        mask_thresh: float = 0.35,
        min_area: int = 50,
        bnd_suppress_thresh: float = 0.5,
    ) -> None:
        self.seed_thresh = seed_thresh
        self.mask_thresh = mask_thresh
        self.min_area = min_area
        self.bnd_suppress_thresh = bnd_suppress_thresh

    def process(
        self,
        sem_prob: np.ndarray,
        bnd_prob: Optional[np.ndarray] = None,
        skl_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert continuous prediction heatmaps into clean instance masks.

        Parameters
        ----------
        sem_prob : (H, W) float32
            Semantic prediction map [0, 1].
        bnd_prob : (H, W) float32, optional
            Boundary prediction map [0, 1].
        skl_prob : (H, W) float32, optional
            Skeleton prediction map [0, 1].

        Returns
        -------
        instances : (K, H, W) uint8
            Array of K instance masks.
        """
        h, w = sem_prob.shape[:2]

        # 1. Candidate mask (permissive threshold)
        cand_mask = (sem_prob > self.mask_thresh).astype(np.uint8)

        if cand_mask.sum() == 0:
            return np.zeros((0, h, w), dtype=np.uint8)

        # 2. Suppress candidate pixels along strong boundaries
        if bnd_prob is not None:
            cand_mask[bnd_prob > self.bnd_suppress_thresh] = 0

        # 3. Extract core high-confidence seeds
        seed_mask = (sem_prob > self.seed_thresh).astype(np.uint8)
        if bnd_prob is not None:
            seed_mask[bnd_prob > 0.4] = 0

        # Connected components on seeds
        num_seeds, seed_labels = cv2.connectedComponents(seed_mask)

        if num_seeds <= 1:
            # Fallback: if no strict seeds, use standard connected components on candidate mask
            num_cand, cand_labels = cv2.connectedComponents(cand_mask)
            valid_insts = []
            for i in range(1, num_cand):
                inst = (cand_labels == i).astype(np.uint8)
                if inst.sum() >= self.min_area:
                    valid_insts.append(inst)
            return np.stack(valid_insts, axis=0) if valid_insts else np.zeros((0, h, w), dtype=np.uint8)

        # 4. Grow seeds into candidate mask via Watershed / Distance Transform Propagation
        dist_transform = cv2.distanceTransform(cand_mask, cv2.DIST_L2, 5)
        watershed_markers = seed_labels.astype(np.int32)
        
        # Grow candidates: map each candidate pixel to its nearest seed label
        # Simple BFS / Dilation propagation
        curr_labels = seed_labels.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        for _ in range(10):  # 10 iterations of controlled seed expansion
            dilated_labels = cv2.dilate(curr_labels.astype(np.uint16), kernel).astype(np.int32)
            curr_labels = np.where((curr_labels == 0) & (cand_mask > 0), dilated_labels, curr_labels)

        # 5. Collect final valid instances exceeding min_area
        valid_insts = []
        unique_labels = np.unique(curr_labels)
        for label in unique_labels:
            if label == 0:
                continue
            inst = (curr_labels == label).astype(np.uint8)
            if inst.sum() >= self.min_area:
                valid_insts.append(inst)

        return np.stack(valid_insts, axis=0) if valid_insts else np.zeros((0, h, w), dtype=np.uint8)
