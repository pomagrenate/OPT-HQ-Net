"""
Inference pipeline: image → cleaned binary masks → submission CSV row.

Applies the full post-processing chain:
    1. Model forward pass (eval mode, no gradients).
    2. Rotated NMS on predicted oriented boxes.
    3. Upscale masks to submission resolution (2048 × 2048).
    4. Morphological cleaning (closing + island filtering).
    5. Fortran-ordered RLE encoding.

Usage
-----
>>> from opt_hq_net.engine import InferencePipeline
>>> pipeline = InferencePipeline(model, device="cuda")
>>> predictions = pipeline.run_on_dataset(test_loader)
>>> submission_df = pipeline.build_submission(predictions)
>>> submission_df.to_csv("submission.csv", index=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opt_hq_net.postprocess.morphological import MorphologicalCleaner
from opt_hq_net.postprocess.rle_encoder import RLEEncoder, build_submission_csv
from opt_hq_net.postprocess.rotated_nms import RotatedNMS


class InferencePipeline:
    """
    End-to-end inference pipeline from batched images to submission CSV.

    Parameters
    ----------
    model : nn.Module
        Trained OPTHQNet in eval state.
    device : str
        'cuda' or 'cpu'.
    nms_iou_threshold : float
        IoU threshold for Rotated NMS.
    score_threshold : float
        Minimum predicted score to keep a detection.
    min_mask_area_px : int
        Minimum connected component area (pixels) to keep.
    submission_size : int
        Final mask resolution (2048 for MAGFiLO).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        nms_iou_threshold: float = 0.40,
        score_threshold: float = 0.05,
        min_mask_area_px: int = 50,
        submission_size: int = 2048,
    ) -> None:
        self.model = model

        # Device validation with sm_60 compatibility check
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                dummy = torch.zeros((1, 1), device=device)
                dummy = dummy + 1.0
                self.device = torch.device(device)
            except Exception:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        self.nms = RotatedNMS(
            iou_threshold=nms_iou_threshold,
            score_threshold=score_threshold,
        )
        self.cleaner = MorphologicalCleaner(min_area_px=min_mask_area_px)
        self.encoder = RLEEncoder(height=submission_size, width=submission_size)
        self.submission_size = submission_size

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_on_batch(
        self, images: torch.Tensor, image_ids: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Run inference on a single batch.

        Parameters
        ----------
        images : Tensor (B, 3, H, W)
        image_ids : list[str]

        Returns
        -------
        dict[image_id, list[np.ndarray (H, W) uint8]]
            Binary masks at ``submission_size`` resolution.
        """
        images = images.to(self.device)
        predictions, _ = self.model(images)

        results: Dict[str, List[np.ndarray]] = {}

        for pred, image_id in zip(predictions, image_ids):
            boxes = pred["boxes"]    # (N, 5) Tensor
            scores = pred["scores"]  # (N,) Tensor
            masks = pred["masks"]    # (N, H, W) Tensor [0,1]

            if len(boxes) == 0:
                results[image_id] = []
                continue

            # Step 1: Rotated NMS on oriented boxes
            keep = self.nms(boxes, scores)
            boxes = boxes[keep]
            masks = masks[keep]

            # Step 2: Upscale masks to submission size
            masks_up = F.interpolate(
                masks.unsqueeze(1).float(),
                size=(self.submission_size, self.submission_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (K, S, S)

            masks_np = masks_up.cpu().numpy()  # float [0,1]

            # Step 3: Morphological cleaning
            cleaned = [self.cleaner(m) for m in masks_np]  # list of uint8 (S, S)

            # Remove empty masks (all-zero after cleaning)
            cleaned = [m for m in cleaned if m.sum() > 0]

            results[image_id] = cleaned

        return results

    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_on_dataset(
        self, data_loader: DataLoader
    ) -> Dict[str, List[np.ndarray]]:
        """
        Run inference over a full dataset loader.

        Parameters
        ----------
        data_loader : DataLoader
            Yields batches with keys 'images' and 'image_ids'.

        Returns
        -------
        dict[image_id, list[np.ndarray]] — all predictions.
        """
        all_predictions: Dict[str, List[np.ndarray]] = {}
        total_batches = len(data_loader)

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        pbar = tqdm(
            data_loader,
            desc="[Inference]",
            disable=not has_tqdm,
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"]
            image_ids = batch["image_ids"]

            batch_preds = self.run_on_batch(images, image_ids)
            all_predictions.update(batch_preds)

            if not has_tqdm and (batch_idx + 1) % 10 == 0:
                print(f"  [Inference] {batch_idx + 1}/{total_batches} batches processed")

        return all_predictions

    # ------------------------------------------------------------------
    def build_submission(
        self, predictions: Dict[str, List[np.ndarray]]
    ) -> "pd.DataFrame":
        """
        Convert predictions dict to a submission-ready DataFrame.

        Parameters
        ----------
        predictions : dict[str, list[np.ndarray]]
            Output of ``run_on_dataset``.

        Returns
        -------
        pd.DataFrame with columns: filament_id, segmentation_rle
        """
        return build_submission_csv(
            predictions,
            height=self.submission_size,
            width=self.submission_size,
        )

    # ------------------------------------------------------------------
    def predict_single_image(
        self, image_np: np.ndarray, image_id: str = "image"
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Convenience method for single-image inference.

        Parameters
        ----------
        image_np : np.ndarray (H, W, 3) float32 in [0,1]
        image_id : str

        Returns
        -------
        masks   : list[np.ndarray (H, W) uint8]
        scores  : list[float]
        """
        # Convert to tensor
        img_t = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).float()
        preds = self.run_on_batch(img_t, [image_id])
        return preds.get(image_id, []), []
