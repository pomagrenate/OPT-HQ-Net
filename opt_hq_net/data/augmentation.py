"""
Solar filament data augmentation pipeline.

Design: ``SolarAugmentation`` is a composable callable that accepts a
(image, masks, boxes) triple and applies a consistent random transform
to all three simultaneously — critical for instance segmentation where
annotation coordinates must stay aligned with the image.

All rotations are full-degree (0° – 360°) to handle arbitrary filament
orientations across the solar disk.

Usage
-----
>>> aug = SolarAugmentation(
...     min_scale=0.8, max_scale=1.2,
...     rotation_degrees=360.0,
...     crop_sizes=[1024, 2048],
... )
>>> image_t, masks_t, boxes_t = aug(image, masks, boxes)
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np


class SolarAugmentation:
    """
    Random spatial augmentation for full-disk H-Alpha solar images.

    Transformations applied (in order):
        1. Random scale jitter  (scale factor uniformly in [min_scale, max_scale]).
        2. Random rotation      (angle uniformly in [0, rotation_degrees)).
        3. Random horizontal flip.
        4. Random vertical flip.
        5. Random crop          (size drawn from crop_sizes).
        6. Resize to target     (target_size × target_size) if specified.

    Parameters
    ----------
    min_scale, max_scale : float
        Scale jitter range.  Values outside [0.5, 2.0] are unusual.
    rotation_degrees : float
        Maximum absolute rotation in degrees.  360 = full-circle.
    crop_sizes : list[int]
        Pool of random crop side lengths.  A size is sampled uniformly.
    target_size : int | None
        If given, the crop is resized to (target_size × target_size)
        after cropping.  Otherwise, the crop is returned as-is.
    flip_horizontal : bool
        Enable random horizontal flip (p = 0.5).
    flip_vertical : bool
        Enable random vertical flip (p = 0.5).
    """

    def __init__(
        self,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        rotation_degrees: float = 360.0,
        crop_sizes: Optional[List[int]] = None,
        target_size: Optional[int] = None,
        flip_horizontal: bool = True,
        flip_vertical: bool = True,
    ) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rotation_degrees = rotation_degrees
        self.crop_sizes = crop_sizes or [1024, 2048]
        self.target_size = target_size
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    # ------------------------------------------------------------------
    def __call__(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the augmentation pipeline.

        Parameters
        ----------
        image : np.ndarray
            Float32 RGB image, shape (H, W, 3), values in [0, 1].
        masks : np.ndarray
            Binary instance masks, shape (N, H, W), dtype uint8/bool.
            N = 0 is acceptable (no filaments in this tile).
        boxes : np.ndarray
            Oriented bounding boxes, shape (N, 5) = [xc, yc, w, h, θ_rad].

        Returns
        -------
        image : np.ndarray  — augmented image (H', W', 3)
        masks : np.ndarray  — augmented masks (N, H', W')
        boxes : np.ndarray  — updated boxes   (N, 5)
        """
        h, w = image.shape[:2]

        # 1. Scale jitter
        scale = random.uniform(self.min_scale, self.max_scale)
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        masks = self._resize_masks(masks, new_h, new_w)
        boxes = self._scale_boxes(boxes, scale)

        # 2. Random rotation
        angle_deg = random.uniform(0, self.rotation_degrees)
        image, masks, boxes = self._rotate(image, masks, boxes, angle_deg)

        # 3. Horizontal flip
        if self.flip_horizontal and random.random() < 0.5:
            image, masks, boxes = self._flip_h(image, masks, boxes)

        # 4. Vertical flip
        if self.flip_vertical and random.random() < 0.5:
            image, masks, boxes = self._flip_v(image, masks, boxes)

        # 5. Random crop
        crop_size = random.choice(self.crop_sizes)
        image, masks, boxes = self._random_crop(image, masks, boxes, crop_size)

        # 6. Optional resize to target
        if self.target_size is not None:
            th, tw = image.shape[:2]
            if th != self.target_size or tw != self.target_size:
                rs = self.target_size / max(th, tw)
                image = cv2.resize(image, (self.target_size, self.target_size))
                masks = self._resize_masks(masks, self.target_size, self.target_size)
                boxes = self._scale_boxes(boxes, rs)

        return image, masks, boxes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
        if masks.ndim == 2:
            masks = masks[np.newaxis]
        n = masks.shape[0]
        out = np.zeros((n, h, w), dtype=masks.dtype)
        for i in range(n):
            out[i] = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        return out

    @staticmethod
    def _scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
        """Scale (xc, yc, w, h) in-place; θ unchanged."""
        if boxes.ndim == 1 or len(boxes) == 0:
            return boxes
        out = boxes.copy()
        out[:, :4] *= scale
        return out

    @staticmethod
    def _rotate(
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        angle_deg: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        rotated_masks = np.zeros_like(masks)
        for i in range(masks.shape[0]):
            rotated_masks[i] = cv2.warpAffine(
                masks[i], M, (w, h), flags=cv2.INTER_NEAREST
            )

        # Rotate box centres + add angle offset
        if len(boxes) > 0:
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            out = boxes.copy()
            # Translate centre to origin, rotate, translate back
            tx = boxes[:, 0] - cx
            ty = boxes[:, 1] - cy
            out[:, 0] = cos_a * tx - sin_a * ty + cx
            out[:, 1] = sin_a * tx + cos_a * ty + cy
            # Accumulate box orientation angle
            out[:, 4] = boxes[:, 4] + angle_rad
            boxes = out

        return image, rotated_masks, boxes

    @staticmethod
    def _flip_h(
        image: np.ndarray, masks: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = image.shape[1]
        image = image[:, ::-1].copy()
        masks = masks[:, :, ::-1].copy()
        if len(boxes) > 0:
            out = boxes.copy()
            out[:, 0] = w - boxes[:, 0]   # flip xc
            out[:, 4] = math.pi - boxes[:, 4]  # mirror θ
            boxes = out
        return image, masks, boxes

    @staticmethod
    def _flip_v(
        image: np.ndarray, masks: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = image.shape[0]
        image = image[::-1].copy()
        masks = masks[:, ::-1].copy()
        if len(boxes) > 0:
            out = boxes.copy()
            out[:, 1] = h - boxes[:, 1]   # flip yc
            out[:, 4] = -boxes[:, 4]      # mirror θ
            boxes = out
        return image, masks, boxes

    @staticmethod
    def _random_crop(
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        size = min(size, h, w)
        y0 = random.randint(0, h - size)
        x0 = random.randint(0, w - size)
        y1, x1 = y0 + size, x0 + size

        image = image[y0:y1, x0:x1]
        masks = masks[:, y0:y1, x0:x1]

        if len(boxes) > 0:
            out = boxes.copy()
            out[:, 0] -= x0
            out[:, 1] -= y0
            # Remove boxes whose centre falls outside the crop
            inside = (
                (out[:, 0] >= 0) & (out[:, 0] < size) &
                (out[:, 1] >= 0) & (out[:, 1] < size)
            )
            out = out[inside]
            masks = masks[inside]
            boxes = out

        return image, masks, boxes
