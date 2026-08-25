"""
Sliced Inference Pipeline (SAHI-style) for high-resolution images.

Tiles large images into overlapping patches, runs model inference on each patch,
and converts predictions back to global image coordinates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class SlicedInference:
    """
    Sliding window sliced inference wrapper.

    Parameters
    ----------
    model : nn.Module
        OPTHQNet model instance.
    patch_size : int
        Tile patch spatial size (e.g. 512).
    overlap : float
        Fractional tile overlap (e.g. 0.25 for 25% overlap).
    score_thresh : float
        Proposal score threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: int = 512,
        overlap: float = 0.25,
        score_thresh: float = 0.05,
    ) -> None:
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.score_thresh = score_thresh
        self.step_size = int(patch_size * (1.0 - overlap))

    @torch.no_grad()
    def predict(
        self,
        image_t: torch.Tensor,  # (3, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Run sliced inference on a single full-resolution image.

        Returns
        -------
        dict with keys 'boxes', 'scores', 'masks'
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        _, h, w = image_t.shape

        if h <= self.patch_size and w <= self.patch_size:
            # Image is already small enough, run direct inference
            preds, _ = self.model(image_t.unsqueeze(0).to(device))
            return preds[0]

        # Calculate grid positions
        y_starts = list(range(0, h - self.patch_size + 1, self.step_size))
        if y_starts[-1] + self.patch_size < h:
            y_starts.append(h - self.patch_size)

        x_starts = list(range(0, w - self.patch_size + 1, self.step_size))
        if x_starts[-1] + self.patch_size < w:
            x_starts.append(w - self.patch_size)

        all_boxes = []
        all_scores = []
        all_masks = []

        for y1 in y_starts:
            for x1 in x_starts:
                y2 = y1 + self.patch_size
                x2 = x1 + self.patch_size

                patch = image_t[:, y1:y2, x1:x2].unsqueeze(0).to(device)
                preds, _ = self.model(patch)
                pred = preds[0]

                boxes = pred["boxes"]   # (N, 5) [xc, yc, w, h, theta] in patch frame
                scores = pred["scores"] # (N,)
                masks = pred["masks"]   # (N, patch_size, patch_size)

                if len(boxes) == 0:
                    continue

                # Filter by score
                keep = scores >= self.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                masks = masks[keep]

                if len(boxes) == 0:
                    continue

                # Shift box center coordinates to full image frame
                boxes_full = boxes.clone()
                boxes_full[:, 0] += x1  # xc shift
                boxes_full[:, 1] += y1  # yc shift

                # Pad patch mask to full image spatial size
                N = len(masks)
                masks_full = torch.zeros((N, h, w), dtype=masks.dtype, device=masks.device)
                masks_full[:, y1:y2, x1:x2] = masks

                all_boxes.append(boxes_full)
                all_scores.append(scores)
                all_masks.append(masks_full)

        if not all_boxes:
            return {
                "boxes": torch.zeros((0, 5), device=device),
                "scores": torch.zeros((0,), device=device),
                "masks": torch.zeros((0, h, w), device=device),
            }

        cat_boxes = torch.cat(all_boxes, dim=0)
        cat_scores = torch.cat(all_scores, dim=0)
        cat_masks = torch.cat(all_masks, dim=0)

        return {
            "boxes": cat_boxes,
            "scores": cat_scores,
            "masks": cat_masks,
        }
