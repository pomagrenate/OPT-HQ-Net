"""
Hierarchical Image Tiler & Tile Stitcher for Filament-HQ.

Extracts fixed 1024x1024 tiles from 2048x2048 observations during training/inference
and stitches tile predictions back into full-resolution instance probability maps
using 2D Gaussian window blending to eliminate seam artifacts.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class ImageTiler:
    """
    Extracts fixed-size overlapping tiles from high-resolution images.
    """

    def __init__(self, tile_size: int = 1024, stride: int = 768) -> None:
        self.tile_size = tile_size
        self.stride = stride

    def extract_tiles(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extract overlapping tiles from an image [H, W, C] or [H, W].

        Returns
        -------
        tiles : list[np.ndarray]
            Extracted tile arrays [tile_size, tile_size, C].
        coords : list[tuple[y1, y2, x1, x2]]
            Bounding coordinates for each tile in original image space.
        """
        h, w = image.shape[:2]
        tiles = []
        coords = []

        y_starts = list(range(0, max(1, h - self.tile_size + 1), self.stride))
        if y_starts[-1] + self.tile_size < h:
            y_starts.append(h - self.tile_size)

        x_starts = list(range(0, max(1, w - self.tile_size + 1), self.stride))
        if x_starts[-1] + self.tile_size < w:
            x_starts.append(w - self.tile_size)

        for y1 in y_starts:
            y2 = min(h, y1 + self.tile_size)
            y1_eff = max(0, y2 - self.tile_size)
            for x1 in x_starts:
                x2 = min(w, x1 + self.tile_size)
                x1_eff = max(0, x2 - self.tile_size)

                tile = image[y1_eff:y2, x1_eff:x2]
                # Pad if image is smaller than tile_size
                if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                    pad_h = self.tile_size - tile.shape[0]
                    pad_w = self.tile_size - tile.shape[1]
                    if tile.ndim == 3:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                    else:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")

                tiles.append(tile)
                coords.append((y1_eff, y2, x1_eff, x2))

        return tiles, coords


class TileStitcher:
    """
    Stitches tile prediction probability maps back into a full-resolution image
    using 2D Gaussian window blending.
    """

    def __init__(self, full_shape: Tuple[int, int], tile_size: int = 1024, num_channels: int = 1) -> None:
        self.h, self.w = full_shape
        self.tile_size = tile_size
        self.num_channels = num_channels

        # Accumulated maps
        self.weight_map = np.zeros((self.h, self.w), dtype=np.float32)
        if num_channels == 1:
            self.prob_map = np.zeros((self.h, self.w), dtype=np.float32)
        else:
            self.prob_map = np.zeros((num_channels, self.h, self.w), dtype=np.float32)

        # Build 2D Gaussian weight window
        self.window = self._gaussian_window_2d(tile_size)

    def add_tile(self, tile_pred: np.ndarray, coord: Tuple[int, int, int, int]) -> None:
        """
        Add a predicted tile map to the full accumulator.

        Parameters
        ----------
        tile_pred : np.ndarray
            Predicted map for tile [tile_size, tile_size] or [C, tile_size, tile_size].
        coord : tuple[y1, y2, x1, x2]
            Tile bounding coordinates in full image.
        """
        y1, y2, x1, x2 = coord
        th = y2 - y1
        tw = x2 - x1
        win = self.window[:th, :tw]

        if self.num_channels == 1:
            if tile_pred.ndim == 3:
                tile_pred = tile_pred.squeeze(0)
            self.prob_map[y1:y2, x1:x2] += tile_pred[:th, :tw] * win
        else:
            if tile_pred.ndim == 2:
                tile_pred = tile_pred[np.newaxis, ...]
            self.prob_map[:, y1:y2, x1:x2] += tile_pred[:, :th, :tw] * win[np.newaxis, ...]

        self.weight_map[y1:y2, x1:x2] += win

    def get_stitched_map(self) -> np.ndarray:
        """Return normalized full-resolution probability map [H, W] or [C, H, W]."""
        eps = 1e-7
        if self.num_channels == 1:
            return np.clip(self.prob_map / (self.weight_map + eps), 0.0, 1.0)
        else:
            return np.clip(self.prob_map / (self.weight_map[np.newaxis, ...] + eps), 0.0, 1.0)

    @staticmethod
    def _gaussian_window_2d(size: int, sigma_scale: float = 0.125) -> np.ndarray:
        """Create 2D Gaussian blending window."""
        sigma = size * sigma_scale
        x = np.arange(size) - (size - 1) / 2.0
        gauss_1d = np.exp(-0.5 * (x / sigma) ** 2)
        gauss_2d = np.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.max()
        return gauss_2d.astype(np.float32)
