"""
Validation Visualizer for Filament-HQ Framework.

Generates 4-panel diagnostic overlay images:
  1. Original Solar Input
  2. Ground Truth Filaments Overlay (colored instances)
  3. Model Predicted Filaments Overlay (colored instances)
  4. Multi-Head Predictions (Semantic, Boundary, Skeleton)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class FilamentVisualizer:
    """
    Validation Diagnostic Visualizer.
    """

    def __init__(self, output_dir: str | Path = "val_visualizations") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_visualization(
        self,
        epoch: int,
        raw_img: np.ndarray,
        gt_instances: np.ndarray,
        pred_sem: np.ndarray,
        pred_bnd: np.ndarray,
        pred_skl: np.ndarray,
        pred_instances: np.ndarray,
        metrics: Dict[str, float],
        save_name: Optional[str] = None,
    ) -> Path:
        """
        Generate and save 4-panel comparison visualization.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        raw_img : (H, W) or (H, W, C) float32/uint8
            Original input image tile.
        gt_instances : (N, H, W) uint8
            Ground truth instance masks.
        pred_sem : (H, W) float32
            Predicted semantic probability map [0, 1].
        pred_bnd : (H, W) float32
            Predicted boundary probability map [0, 1].
        pred_skl : (H, W) float32
            Predicted skeleton probability map [0, 1].
        pred_instances : (K, H, W) uint8
            Predicted instance masks after post-processing.
        metrics : dict
            Current validation metrics (Dice, PQ, GT count, Pred count).
        save_name : str, optional
            Custom filename override.
        """
        h, w = pred_sem.shape[:2]

        # Prepare normalized background RGB image
        if raw_img.ndim == 2:
            base_rgb = cv2.cvtColor((np.clip(raw_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif raw_img.ndim == 3 and raw_img.shape[2] == 4:
            base_rgb = cv2.cvtColor((np.clip(raw_img[:, :, 0], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = (np.clip(raw_img, 0, 1) * 255).astype(np.uint8)

        # 1. Build GT Colored Overlay
        gt_overlay = base_rgb.copy()
        if len(gt_instances) > 0:
            gt_overlay = self._draw_instance_overlay(gt_overlay, gt_instances)

        # 2. Build Pred Colored Overlay
        pred_overlay = base_rgb.copy()
        if len(pred_instances) > 0:
            pred_overlay = self._draw_instance_overlay(pred_overlay, pred_instances)

        # 3. Build Multi-Head Composite RGB (R: Semantic, G: Boundary, B: Skeleton)
        heads_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        heads_rgb[:, :, 0] = (np.clip(pred_sem, 0, 1) * 255).astype(np.uint8)
        heads_rgb[:, :, 1] = (np.clip(pred_bnd, 0, 1) * 255).astype(np.uint8)
        heads_rgb[:, :, 2] = (np.clip(pred_skl, 0, 1) * 255).astype(np.uint8)

        # Plot 2x2 Grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=150)
        fig.suptitle(
            f"Epoch {epoch:03d} Validation Audit | Dice: {metrics.get('mean_dice', 0):.4f} | "
            f"PQ: {metrics.get('PQ', 0):.4f} | GT: {len(gt_instances)} | Pred: {len(pred_instances)}",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1: Raw Input
        axes[0, 0].imshow(base_rgb)
        axes[0, 0].set_title("1. Raw Solar Tile Input (1024x1024)", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        # Panel 2: GT Overlay
        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title(f"2. GT Instance Annotations ({len(gt_instances)} filaments)", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

        # Panel 3: Pred Overlay
        axes[1, 0].imshow(pred_overlay)
        axes[1, 0].set_title(f"3. Model Predictions ({len(pred_instances)} instances)", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        # Panel 4: Multi-Head Maps (R: Sem, G: Bnd, B: Skl)
        axes[1, 1].imshow(heads_rgb)
        axes[1, 1].set_title("4. Multi-Head (Red: Semantic, Green: Boundary, Blue: Skeleton)", fontsize=11, fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout()

        out_path = self.output_dir / (save_name or f"epoch_{epoch:03d}_val.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        return out_path

    @staticmethod
    def _draw_instance_overlay(base_rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Draw colored instance polygon masks and boundaries on top of base RGB.
        """
        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(max(len(masks), 100), 3), dtype=np.uint8)
        overlay = base_rgb.copy()

        for idx, mask in enumerate(masks):
            if mask.sum() == 0:
                continue
            color = colors[idx % len(colors)].tolist()
            colored_mask = np.zeros_like(base_rgb, dtype=np.uint8)
            colored_mask[mask > 0] = color

            # Blend mask
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

            # Draw boundary contour
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay
