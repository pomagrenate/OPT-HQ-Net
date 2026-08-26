"""
Solar Physical-Aware Preprocessor for H-Alpha Observations (Filament-HQ).

Generates a 4-channel representation for high-resolution filament segmentation:
  Channel 0: Normalized raw H-alpha intensity [p1, p99] -> [0, 1]
  Channel 1: High-pass local contrast (I - GaussianBlur(I))
  Channel 2: Dark-structure response (Black-Hat morph filter: Closing(I) - I)
  Channel 3: Solar Disk Normalized Radial Distance Map (r = d_center / R_disk)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


class SolarPhysicalPreprocessor:
    """
    4-channel physical-aware preprocessor for H-alpha solar disk observations.Stateless callable suitable for multiprocessing PyTorch DataLoaders.
    """

    def __init__(
        self,
        p_min: float = 1.0,
        p_max: float = 99.0,
        blur_kernel_size: int = 31,
        morph_kernel_size: int = 15,
    ) -> None:
        self.p_min = p_min
        self.p_max = p_max
        self.blur_ksize = blur_kernel_size | 1  # ensure odd
        self.morph_ksize = morph_kernel_size | 1

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Process a 2D grayscale or RGB image into a 4-channel float32 array [H, W, 4].

        Parameters
        ----------
        image : np.ndarray
            Input H-alpha image (H, W) or (H, W, 3), uint8 or float32.

        Returns
        -------
        np.ndarray
            4-channel preprocessed array, shape (H, W, 4), float32 in [0, 1].
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
        else:
            gray = image.copy()

        if gray.dtype != np.float32:
            gray = gray.astype(np.float32)

        # Ensure input range [0, 1] if uint8
        if gray.max() > 1.0:
            gray = gray / 255.0

        # --- Channel 0: Percentile-Normalized Raw H-Alpha ---
        v_min, v_max = np.percentile(gray, (self.p_min, self.p_max))
        if v_max > v_min:
            c0 = np.clip((gray - v_min) / (v_max - v_min), 0.0, 1.0)
        else:
            c0 = gray.copy()

        # --- Channel 1: High-Pass Local Contrast ---
        blurred = cv2.GaussianBlur(c0, (self.blur_ksize, self.blur_ksize), 0)
        c1 = c0 - blurred
        # Normalize local contrast to [0, 1]
        c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-7)

        # --- Channel 2: Dark-Structure Response (Black-Hat Morphological Filter) ---
        # Black-Hat = Closing(I) - I (highlights dark elongated features on bright background)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_ksize, self.morph_ksize))
        closing = cv2.morphologyEx(c0, cv2.MORPH_CLOSE, kernel)
        c2 = closing - c0
        c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-7)

        # --- Channel 3: Solar Disk Positional Distance Map ---
        h, w = gray.shape[:2]
        c3 = self._compute_solar_radial_distance(c0, h, w)

        ch4 = np.stack([c0, c1, c2, c3], axis=-1).astype(np.float32)
        return ch4

    def _compute_solar_radial_distance(self, c0: np.ndarray, h: int, w: int) -> np.ndarray:
        """Derive radial distance from detected solar disk center normalized by disk radius."""
        try:
            # Downsample for fast circle detection
            scale = 512.0 / max(h, w)
            small = cv2.resize((c0 * 255).astype(np.uint8), (int(w * scale), int(h * scale)))
            small_blur = cv2.medianBlur(small, 9)

            circles = cv2.HoughCircles(
                small_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=small.shape[0] // 2,
                param1=50,
                param2=30,
                minRadius=int(small.shape[0] * 0.3),
                maxRadius=int(small.shape[0] * 0.5),
            )

            if circles is not None and len(circles) > 0:
                circle = circles[0][0]
                cx, cy, r = circle[0] / scale, circle[1] / scale, circle[2] / scale
            else:
                cx, cy, r = w / 2.0, h / 2.0, min(h, w) * 0.45
        except Exception:
            cx, cy, r = w / 2.0, h / 2.0, min(h, w) * 0.45

        # Compute radial distance grid
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_map = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / max(r, 1.0)
        dist_map = np.clip(dist_map, 0.0, 1.5).astype(np.float32)
        return dist_map
