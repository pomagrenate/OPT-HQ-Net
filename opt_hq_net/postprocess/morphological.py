"""
Morphological mask cleaning and island filtering.

Post-processing steps applied to raw predicted binary masks:
    1. Threshold probability map to binary mask (default: 0.5).
    2. Remove disconnected components smaller than ``min_area_px`` pixels.
    3. Apply 3×3 morphological closing to bridge faint spine gaps.

Usage
-----
>>> cleaner = MorphologicalCleaner(min_area_px=50, closing_kernel_size=3)
>>> clean_mask = cleaner(prob_map)   # np.ndarray uint8 (H, W)
"""

from __future__ import annotations

import cv2
import numpy as np


class MorphologicalCleaner:
    """
    Clean a predicted binary mask by removing noise islands and
    closing faint boundary gaps.

    Parameters
    ----------
    min_area_px : int
        Connected components with fewer pixels than this threshold are
        discarded as background noise.
    closing_kernel_size : int
        Side length of the square structuring element for morphological
        closing (bridges small gaps in the filament spine).
    threshold : float
        Probability threshold used to binarise the input probability map.
    """

    def __init__(
        self,
        min_area_px: int = 50,
        closing_kernel_size: int = 3,
        threshold: float = 0.5,
    ) -> None:
        self.min_area_px = min_area_px
        self.closing_kernel_size = closing_kernel_size
        self.threshold = threshold

        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (closing_kernel_size, closing_kernel_size),
        )

    # ------------------------------------------------------------------
    def __call__(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply cleaning pipeline to a probability map.

        Parameters
        ----------
        prob_map : np.ndarray (H, W)
            Float probability map in [0, 1].

        Returns
        -------
        np.ndarray (H, W) uint8
            Cleaned binary mask with values in {0, 1}.
        """
        # Step 1: Threshold
        binary = (prob_map >= self.threshold).astype(np.uint8)

        # Step 2: Morphological closing (bridge faint spine gaps)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)

        # Step 3: Remove small disconnected components
        cleaned = self._remove_small_components(closed)

        return cleaned

    # ------------------------------------------------------------------
    def _remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than ``min_area_px``."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        output = np.zeros_like(binary)
        for label_id in range(1, num_labels):           # 0 is background
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= self.min_area_px:
                output[labels == label_id] = 1

        return output

    # ------------------------------------------------------------------
    def process_batch(self, prob_maps: np.ndarray) -> np.ndarray:
        """
        Apply cleaning to a batch of probability maps.

        Parameters
        ----------
        prob_maps : np.ndarray (N, H, W)

        Returns
        -------
        np.ndarray (N, H, W) uint8
        """
        return np.stack([self(p) for p in prob_maps], axis=0)
