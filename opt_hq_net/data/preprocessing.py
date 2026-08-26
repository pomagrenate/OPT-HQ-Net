"""
Solar H-Alpha image pre-processing utilities.

Classes
-------
SolarDiskMask
    Generates a binary circular mask isolating the solar disk from
    deep-space background pixels.

CLAHEPreprocessor
    Applies Contrast Limited Adaptive Histogram Equalisation (CLAHE)
    to enhance subtle contrast gradients in ground-based H-Alpha
    observations, followed by normalisation to [0, 1].

Both classes are stateless callables — safe to share across DataLoader
workers.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Solar disk masking
# ---------------------------------------------------------------------------

class SolarDiskMask:
    """
    Create a circular binary mask that isolates the solar disk.

    The mask is derived by finding the largest bright circle in the image
    (Hough circle transform on a downsampled copy for speed).  Deep-space
    background pixels are set to zero so the model never generates false
    positive detections outside the disk.

    Parameters
    ----------
    margin_px : int
        Safety margin (pixels) subtracted from the detected disk radius to
        avoid clipping at the limb.

    Examples
    --------
    >>> masker = SolarDiskMask(margin_px=10)
    >>> masked_img = masker(image)   # np.ndarray HxWx3, values in [0,1]
    """

    def __init__(self, margin_px: int = 10) -> None:
        self.margin_px = margin_px

    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the solar disk mask to an image.

        Parameters
        ----------
        image : np.ndarray
            RGB or grayscale image, float32 in [0, 1] or uint8 in [0, 255].
            Shape: (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Image with pixels outside the solar disk set to 0.  Same dtype
            and shape as input.
        """
        h, w = image.shape[:2]
        mask = self._detect_disk_mask(image, h, w)

        if image.ndim == 3:
            masked = image * mask[:, :, np.newaxis]
        else:
            masked = image * mask

        return masked

    # ------------------------------------------------------------------
    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Return the boolean disk mask (H × W) without applying it."""
        h, w = image.shape[:2]
        return self._detect_disk_mask(image, h, w).astype(bool)

    # ------------------------------------------------------------------
    def _detect_disk_mask(
        self, image: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Hough-based disk detection → binary float32 mask."""
        # Convert to uint8 grayscale for Hough
        if image.dtype != np.uint8:
            gray = (image * 255).astype(np.uint8)
        else:
            gray = image.copy()

        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        # Downsample for speed
        scale = 512 / max(h, w)
        small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        small = cv2.medianBlur(small, 5)

        circles = cv2.HoughCircles(
            small,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=small.shape[0] // 2,
            param1=50,
            param2=30,
            minRadius=int(small.shape[0] * 0.3),
            maxRadius=int(small.shape[0] * 0.6),
        )

        mask = np.zeros((h, w), dtype=np.float32)

        if circles is not None:
            cx, cy, r = circles[0, 0]
            # Scale back to original resolution
            cx = int(cx / scale)
            cy = int(cy / scale)
            r = int(r / scale) - self.margin_px
            cv2.circle(mask, (cx, cy), max(r, 1), 1.0, thickness=-1)
        else:
            # Fallback: assume disk fills 90 % of the image
            cx, cy = w // 2, h // 2
            r = int(min(h, w) * 0.45) - self.margin_px
            cv2.circle(mask, (cx, cy), r, 1.0, thickness=-1)

        return mask


# ---------------------------------------------------------------------------
# CLAHE normalisation
# ---------------------------------------------------------------------------

class CLAHEPreprocessor:
    """
    Contrast Limited Adaptive Histogram Equalisation for solar H-Alpha images.

    Converts a raw observation (14-bit FITS or 8-bit PNG) into a
    contrast-enhanced, normalised float32 image ready for the neural
    network backbone.

    Pipeline
    --------
    1. Clip & scale to uint8.
    2. Apply CLAHE tile-by-tile.
    3. Repeat across R/G/B channels (solar images are often grayscale
       replicated into three channels).
    4. Normalise to [0, 1].
    5. Optionally stack into three identical channels (backbone expects
       3-channel input).

    Parameters
    ----------
    clip_limit : float
        CLAHE contrast limit. Higher values give more aggressive enhancement.
    tile_grid_size : tuple[int, int]
        Number of CLAHE tiles along (height, width).
    bit_depth : int
        Input image bit depth (8 for PNG, 14 for raw FITS).

    Examples
    --------
    >>> preprocessor = CLAHEPreprocessor(clip_limit=2.0)
    >>> tensor = preprocessor(raw_image)  # torch.Tensor, shape (3, H, W)
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        bit_depth: int = 8,
    ) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.bit_depth = bit_depth

    # ------------------------------------------------------------------
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Pre-process a single solar image.

        Parameters
        ----------
        image : np.ndarray
            Raw image array.  Shape: (H, W) or (H, W, C).
            Accepts any integer or float dtype; values are clipped and
            rescaled internally.

        Returns
        -------
        np.ndarray
            Float32 array, shape (H, W, 3), values in [0, 1].
        """
        gray = self._to_gray_uint8(image)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        enhanced = clahe.apply(gray)

        # Normalise to float32 [0, 1]
        normalised = enhanced.astype(np.float32) / 255.0

        # Replicate to 3 channels (backbone expects RGB)
        rgb = np.stack([normalised, normalised, normalised], axis=-1)
        return rgb

    # ------------------------------------------------------------------
    def _to_gray_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convert arbitrary input to uint8 grayscale."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Handle 14-bit or higher
        max_val = 2**self.bit_depth - 1
        image = np.clip(image, 0, max_val)
        image = (image.astype(np.float64) / max_val * 255.0).astype(np.uint8)
        return image
