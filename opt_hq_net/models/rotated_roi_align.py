"""
Rotated RoI Align — pure PyTorch implementation.

Extracts a fixed-size (output_size × output_size) feature crop aligned
to an arbitrary-angle oriented bounding box, using bilinear grid sampling
(``torch.nn.functional.grid_sample``).

How it works
------------
For each oriented proposal (xc, yc, w, h, θ):

1. Build a 2×3 affine matrix that maps from the output grid coordinates
   ([-1, 1]²) to the rotated box region in the feature map's pixel space.
2. Use ``F.affine_grid`` to generate the sampling grid.
3. Use ``F.grid_sample`` with bilinear interpolation to extract features.

This physically decouples the feature crop from adjacent overlapping
filaments that fall outside the rotated box boundaries.

Usage
-----
>>> roi_align = RotatedRoIAlign(output_size=28, spatial_scale=1/4)
>>> crops = roi_align(feature_map_P2, boxes, image_size=(2048, 2048))
# crops: (N_total, C, 28, 28)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatedRoIAlign(nn.Module):
    """
    Rotated Region-of-Interest Align (pure PyTorch).

    Extracts ``output_size × output_size`` feature crops aligned along the
    natural orientation axis of each oriented bounding box.

    Parameters
    ----------
    output_size : int
        Side length of the square output feature map (default 28).
    spatial_scale : float
        Ratio from image space to feature map space (e.g. 1/4 for P2).
    sampling_ratio : int
        Not used directly here (kept for API compatibility with
        torchvision-style RoIAlign).  Bilinear interpolation is used.

    Input
    -----
    feature_map : Tensor (B, C, H_feat, W_feat)
    boxes       : list[Tensor]  — one per image, each (N_i, 5+1)
                  columns: [xc, yc, w, h, θ_rad, score]
                  Coordinates are in *image* pixel space.

    Output
    ------
    crops : Tensor (N_total, C, output_size, output_size)
        Concatenated crops from all images.
    batch_idx : Tensor (N_total,)
        Which image in the batch each crop belongs to.
    """

    def __init__(
        self,
        output_size: int = 28,
        spatial_scale: float = 0.25,
        sampling_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    # ------------------------------------------------------------------
    def forward(
        self,
        feature_map: torch.Tensor,
        boxes_per_image: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract rotated crops from a single FPN level."""
        B, C, H_f, W_f = feature_map.shape
        device = feature_map.device
        out_s = self.output_size

        all_crops: List[torch.Tensor] = []
        all_batch_idx: List[torch.Tensor] = []

        for img_idx, boxes in enumerate(boxes_per_image):
            if len(boxes) == 0:
                continue

            feat = feature_map[img_idx].unsqueeze(0)   # (1, C, H_f, W_f)
            # Scale boxes from image space to feature map space
            s = self.spatial_scale
            # boxes: (N, 5+) — [xc, yc, w, h, θ, ...]
            xc = boxes[:, 0] * s
            yc = boxes[:, 1] * s
            bw = boxes[:, 2] * s
            bh = boxes[:, 3] * s
            theta = boxes[:, 4]

            N = len(boxes)

            # Build affine matrices: (N, 2, 3)
            # Maps normalised output coords [-1,1] → feature map coords
            cos_t = torch.cos(theta)   # (N,)
            sin_t = torch.sin(theta)

            # The affine matrix transforms output sampling grid to feature coords.
            # Output grid spans [-1,1]; we want to map to the rotated box region.
            # Affine: x_feat = A * x_out + t
            # where A is the rotation-scale matrix and t is the centre.
            #
            # In normalised coordinates for grid_sample the formula is:
            #   x_norm = (x_feat / (W_f/2)) - 1
            # so we compose accordingly.

            A = torch.zeros(N, 2, 3, device=device)

            # Scale factors from [-1,1] to feature pixel space
            sx = bw / 2.0         # half-width in feature px
            sy = bh / 2.0         # half-height

            # Rotation + scale, then normalise to [-1,1]
            A[:, 0, 0] = cos_t * sx / (W_f / 2.0)
            A[:, 0, 1] = -sin_t * sy / (W_f / 2.0)
            A[:, 0, 2] = xc / (W_f / 2.0) - 1.0      # translate centre
            A[:, 1, 0] = sin_t * sx / (H_f / 2.0)
            A[:, 1, 1] = cos_t * sy / (H_f / 2.0)
            A[:, 1, 2] = yc / (H_f / 2.0) - 1.0

            # Generate sampling grid: (N, out_s, out_s, 2)
            grid = F.affine_grid(A, (N, C, out_s, out_s), align_corners=False)

            # Flatten grid into (1, N * out_s, out_s, 2) to sample directly from single feat (1, C, H_f, W_f)
            # This avoids expanding feature map to (N, C, H_f, W_f) which consumes gigabytes of VRAM
            grid_reshaped = grid.reshape(1, N * out_s, out_s, 2)

            # Sample without expanding feature map
            crops_sampled = F.grid_sample(
                feat, grid_reshaped, mode="bilinear", align_corners=False, padding_mode="zeros"
            )  # (1, C, N * out_s, out_s)

            # Reshape back to (N, C, out_s, out_s)
            crops = crops_sampled.reshape(N, C, out_s, out_s)

            all_crops.append(crops)
            all_batch_idx.append(torch.full((N,), img_idx, dtype=torch.long, device=device))

        if not all_crops:
            return (
                torch.zeros(0, feature_map.shape[1], out_s, out_s, device=device, dtype=feature_map.dtype),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        return torch.cat(all_crops, dim=0), torch.cat(all_batch_idx, dim=0)


# ---------------------------------------------------------------------------
# Multi-scale Rotated RoI Align (selects FPN level by box area)
# ---------------------------------------------------------------------------

class MultiScaleRotatedRoIAlign(nn.Module):
    """
    Apply ``RotatedRoIAlign`` at the appropriate FPN level for each box,
    following the canonical FPN scale assignment:

        level = clip(floor(k0 + log2(sqrt(w*h) / canonical_scale)), lvl_min, lvl_max)

    Parameters
    ----------
    output_size : int
        RoI crop size.
    canonical_scale : float
        Canonical object scale (pixels) anchoring the level assignment.
    canonical_level : int
        FPN level index for the canonical scale (usually 4 for P4).
    level_min, level_max : int
        Clipping bounds for the level index.
    """

    def __init__(
        self,
        output_size: int = 28,
        canonical_scale: float = 224.0,
        canonical_level: int = 4,
        level_min: int = 2,
        level_max: int = 5,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.level_min = level_min
        self.level_max = level_max

        # One RoIAlign per FPN level
        self._roi_aligns = nn.ModuleDict({
            f"P{lvl}": RotatedRoIAlign(
                output_size=output_size,
                spatial_scale=1.0 / (2 ** lvl),
            )
            for lvl in range(level_min, level_max + 1)
        })

    # ------------------------------------------------------------------
    def _assign_level(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute FPN level index for each box based on box area."""
        areas = boxes[:, 2] * boxes[:, 3]   # w * h
        scale = torch.sqrt(areas)
        target_lvl = torch.floor(
            self.canonical_level + torch.log2(scale / self.canonical_scale + 1e-8)
        ).long()
        return target_lvl.clamp(self.level_min, self.level_max)

    # ------------------------------------------------------------------
    def forward(
        self,
        features: dict,
        boxes_per_image: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract crops from appropriate FPN levels.

        Parameters
        ----------
        features : dict[str, Tensor]
            FPN feature pyramid.
        boxes_per_image : list[Tensor]
            Oriented proposals per image (N_i, 5+1).

        Returns
        -------
        crops : Tensor (N_total, C, output_size, output_size)
        batch_idx : Tensor (N_total,)
        """
        all_crops: List[torch.Tensor] = []
        all_batch_idx: List[torch.Tensor] = []
        device = next(iter(features.values())).device

        for img_idx, boxes in enumerate(boxes_per_image):
            if len(boxes) == 0:
                continue

            levels = self._assign_level(boxes)

            img_crops: List[torch.Tensor] = []
            for lvl in range(self.level_min, self.level_max + 1):
                key = f"P{lvl}"
                if key not in features:
                    continue
                mask = levels == lvl
                if not mask.any():
                    continue
                level_boxes = boxes[mask]
                # Wrap as per-image list for single level
                crops, _ = self._roi_aligns[key](
                    features[key],
                    [level_boxes],
                )
                # Re-index into original box ordering (preserve dtype for AMP FP16)
                img_crops_ordered = torch.zeros(
                    len(boxes),
                    crops.shape[1],
                    self.output_size,
                    self.output_size,
                    device=device,
                    dtype=crops.dtype,
                )
                img_crops_ordered[mask] = crops
                img_crops.append((mask, img_crops_ordered))

            if img_crops:
                # Merge level crops: take the one non-zero entry per box
                first_crops_dtype = img_crops[0][1].dtype
                merged = torch.zeros(
                    len(boxes),
                    list(features.values())[0].shape[1],
                    self.output_size,
                    self.output_size,
                    device=device,
                    dtype=first_crops_dtype,
                )
                for m, c in img_crops:
                    merged[m] = c[m]
                all_crops.append(merged)
                all_batch_idx.append(
                    torch.full((len(boxes),), img_idx, dtype=torch.long, device=device)
                )

        if not all_crops:
            first_feat = next(iter(features.values()))
            C = first_feat.shape[1]
            return (
                torch.zeros(0, C, self.output_size, self.output_size, device=device, dtype=first_feat.dtype),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        return torch.cat(all_crops, dim=0), torch.cat(all_batch_idx, dim=0)
