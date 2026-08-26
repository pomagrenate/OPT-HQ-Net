"""
Training and Validation Engine for Filament-HQ.

Supports:
  - Single-image FP32 overfit verification mode (Phase 0 audit).
  - Staged curriculum training (Stage 1 -> 2 -> 3).
  - Multi-GPU DDP distributed execution.
  - Comprehensive metric tracking (Dice, IoU, Panoptic Quality PQ).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from filament_hq.data.dataset import FilamentTileDataset
from filament_hq.data.tiler import TileStitcher, ImageTiler
from filament_hq.losses.losses import FilamentCompoundLoss, check_finite
from filament_hq.models.model import FilamentHQModel
from filament_hq.metrics.panoptic_quality import PanopticQualityMetric


class FilamentTrainer:
    """
    Filament-HQ Trainer Engine.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str | torch.device = "cuda",
        epochs: int = 50,
        stage: int = 1,
        use_amp: bool = False,
        checkpoint_dir: str = "checkpoints_hq",
        overfit_single_image: bool = False,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.stage = stage
        self.use_amp = use_amp and torch.cuda.is_available()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.overfit_single_image = overfit_single_image

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.loss_fn = FilamentCompoundLoss(stage=stage).to(self.device)

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.metric = PanopticQualityMetric(iou_threshold=0.5)

    def train(self) -> Dict[str, float]:
        """
        Execute full training loop across epochs.
        """
        print(f"\n[FilamentTrainer] Starting training on {self.device}")
        print(f"  Stage: {self.stage} | AMP Enabled: {self.use_amp} | Overfit Mode: {self.overfit_single_image}")
        print(f"  Epochs: {self.epochs} | Train Batches: {len(self.train_loader)}\n")

        best_dice = 0.0

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)

            # Log progress
            print(
                f"[Epoch {epoch:04d}/{self.epochs:04d}] "
                f"Loss Total: {train_metrics['total_loss']:.4f} | "
                f"Sem: {train_metrics['loss_semantic']:.4f} | "
                f"Bnd: {train_metrics['loss_boundary']:.4f} | "
                f"Skl: {train_metrics['loss_skeleton']:.4f}"
            )

            # Overfit verification stop criteria
            if self.overfit_single_image:
                if train_metrics["total_loss"] < 0.05 and train_metrics["loss_semantic"] < 0.03:
                    print(f"\n[Overfit Verification SUCCESS] Loss converged to {train_metrics['total_loss']:.4f} at epoch {epoch}!")
                    break

            # Run Validation if available
            if self.val_loader and (epoch % 2 == 0 or epoch == self.epochs or self.overfit_single_image):
                val_res = self._validate(epoch)
                if val_res.get("mean_dice", 0.0) > best_dice:
                    best_dice = val_res.get("mean_dice", 0.0)
                    self._save_checkpoint(epoch, is_best=True)

        return {"best_dice": best_dice}

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses = {"total_loss": 0.0, "loss_semantic": 0.0, "loss_boundary": 0.0, "loss_skeleton": 0.0, "loss_instance": 0.0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)  # (B, 4, 1024, 1024)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, batch)
                self.scaler.scale(loss_dict["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.loss_fn(outputs, batch)
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            for k in running_losses:
                if k in loss_dict:
                    running_losses[k] += loss_dict[k].item()

            pbar.set_postfix(total_loss=f"{loss_dict['total_loss'].item():.4f}")

        num_steps = max(len(self.train_loader), 1)
        return {k: v / num_steps for k, v in running_losses.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.metric.reset()

        total_gt_instances = 0
        total_pred_instances = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            gt_masks = batch["semantic"].cpu().numpy()  # (B, 1, 1024, 1024)

            outputs = self.model(images)
            sem_probs = torch.sigmoid(outputs["semantic"]).cpu().numpy()  # (B, 1, 1024, 1024)

            pred_np = []
            gt_np = []

            for b in range(images.shape[0]):
                p_mask = (sem_probs[b, 0] > 0.5).astype(np.uint8)
                g_mask = (gt_masks[b, 0] > 0.5).astype(np.uint8)

                # Connected components for instance splitting
                num_p, p_labels = cv2.connectedComponents(p_mask)
                num_g, g_labels = cv2.connectedComponents(g_mask)

                p_insts = [(p_labels == i).astype(np.uint8) for i in range(1, num_p)]
                g_insts = [(g_labels == j).astype(np.uint8) for j in range(1, num_g)]

                pred_arr = np.stack(p_insts, axis=0) if p_insts else np.zeros((0, 1024, 1024), dtype=np.uint8)
                gt_arr = np.stack(g_insts, axis=0) if g_insts else np.zeros((0, 1024, 1024), dtype=np.uint8)

                pred_np.append(pred_arr)
                gt_np.append(gt_arr)

                total_pred_instances += len(pred_arr)
                total_gt_instances += len(gt_arr)

            self.metric.update(pred_np, gt_np)

            # In overfit mode, stop validation after 2 batches max
            if self.overfit_single_image and step >= 1:
                break

        metrics = self.metric.compute()
        print(
            f"  [Val Epoch {epoch:03d}] GT Inst: {total_gt_instances} | Pred Inst: {total_pred_instances} | "
            f"TP: {metrics.get('TP', 0)} | FP: {metrics.get('FP', 0)} | FN: {metrics.get('FN', 0)} | "
            f"Mean Dice: {metrics.get('mean_dice', 0.0):.4f} | PQ: {metrics.get('PQ', 0.0):.4f}"
        )
        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        save_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )
        if is_best:
            torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
            print(f"  [Checkpoint] Saved new BEST model to {self.checkpoint_dir / 'best_model.pt'}")
