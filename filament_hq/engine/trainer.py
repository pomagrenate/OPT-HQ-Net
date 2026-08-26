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
from filament_hq.engine.ema import ModelEMA
from filament_hq.losses.losses import FilamentCompoundLoss, check_finite
from filament_hq.metrics.panoptic_quality import PanopticQualityMetric
from filament_hq.models.model import FilamentHQModel
from filament_hq.postprocessing.grouping import FilamentPostProcessor
from filament_hq.visualization.visualizer import FilamentVisualizer


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
        self.ema = ModelEMA(self.model, decay=0.999, device=str(self.device))

        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.metric = PanopticQualityMetric(iou_threshold=0.5)
        self.postprocessor = FilamentPostProcessor(seed_thresh=0.65, mask_thresh=0.35, min_area=50)
        self.visualizer = FilamentVisualizer(output_dir=self.checkpoint_dir / "val_visualizations")

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
        t_data_start = time.time()

        for step, batch in enumerate(pbar):
            t_data = time.time() - t_data_start

            t_fwd_start = time.time()
            images = batch["image"].to(self.device, non_blocking=True)  # (B, 4, 1024, 1024)
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.use_amp:
                try:
                    autocast_cm = torch.amp.autocast("cuda")
                except AttributeError:
                    autocast_cm = torch.cuda.amp.autocast()

                with autocast_cm:
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, batch)
                t_fwd = time.time() - t_fwd_start

                t_bwd_start = time.time()
                self.scaler.scale(loss_dict["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.model)
                t_bwd = time.time() - t_bwd_start
            else:
                outputs = self.model(images)
                loss_dict = self.loss_fn(outputs, batch)
                t_fwd = time.time() - t_fwd_start

                t_bwd_start = time.time()
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.ema.update(self.model)
                t_bwd = time.time() - t_bwd_start

            if epoch == 1 and step == 0:
                print(f"\n[PROFILE Step 1] Data Load: {t_data*1000:.1f}ms | Forward: {t_fwd*1000:.1f}ms | Backward: {t_bwd*1000:.1f}ms")

            for k in running_losses:
                if k in loss_dict:
                    running_losses[k] += loss_dict[k].item()

            pbar.set_postfix(total_loss=f"{loss_dict['total_loss'].item():.4f}")
            t_data_start = time.time()

        num_steps = max(len(self.train_loader), 1)
        return {k: v / num_steps for k, v in running_losses.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        eval_model = self.ema.module
        eval_model.eval()
        self.metric.reset()

        total_gt_instances = 0
        total_pred_instances = 0
        total_coco_gt = 0

        first_sample_data = None

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            gt_semantic = batch["semantic"].cpu().numpy()  # (B, 1, 1024, 1024)

            outputs = eval_model(images)
            sem_probs = torch.sigmoid(outputs["semantic"]).cpu().numpy()  # (B, 1, 1024, 1024)
            bnd_probs = torch.sigmoid(outputs["boundary"]).cpu().numpy() if "boundary" in outputs else None
            skl_probs = torch.sigmoid(outputs["skeleton"]).cpu().numpy() if "skeleton" in outputs else None

            pred_np = []
            gt_np = []

            for b in range(images.shape[0]):
                # 1. Extract GT Instance Masks (prefer exact COCO instance list if available)
                if "instance_masks" in batch and isinstance(batch["instance_masks"], list):
                    gt_arr = batch["instance_masks"][b]
                    if isinstance(gt_arr, torch.Tensor):
                        gt_arr = gt_arr.cpu().numpy()
                    total_coco_gt += len(gt_arr)
                else:
                    g_mask = (gt_semantic[b, 0] > 0.5).astype(np.uint8)
                    num_g, g_labels = cv2.connectedComponents(g_mask)
                    g_insts = [(g_labels == j).astype(np.uint8) for j in range(1, num_g)]
                    gt_arr = np.stack(g_insts, axis=0) if g_insts else np.zeros((0, 1024, 1024), dtype=np.uint8)

                # 2. Hysteresis Post-Processing Grouping for Predicted Instances
                b_bnd = bnd_probs[b, 0] if bnd_probs is not None else None
                b_skl = skl_probs[b, 0] if skl_probs is not None else None

                pred_arr = self.postprocessor.process(
                    sem_prob=sem_probs[b, 0],
                    bnd_prob=b_bnd,
                    skl_prob=b_skl,
                )

                pred_np.append(pred_arr)
                gt_np.append(gt_arr)

                total_pred_instances += len(pred_arr)
                total_gt_instances += len(gt_arr)

                # Cache first sample for diagnostic visualization
                if first_sample_data is None:
                    raw_img = images[b].cpu().numpy().transpose(1, 2, 0)
                    first_sample_data = (
                        raw_img,
                        gt_arr,
                        sem_probs[b, 0],
                        b_bnd if b_bnd is not None else np.zeros_like(sem_probs[b, 0]),
                        b_skl if b_skl is not None else np.zeros_like(sem_probs[b, 0]),
                        pred_arr,
                    )

            self.metric.update(pred_np, gt_np)

            if self.overfit_single_image and step >= 1:
                break

        metrics = self.metric.compute()

        # Save diagnostic 4-panel visualization
        if first_sample_data is not None:
            raw_img, gt_arr, p_sem, p_bnd, p_skl, p_inst = first_sample_data
            vis_path = self.visualizer.generate_visualization(
                epoch=epoch,
                raw_img=raw_img,
                gt_instances=gt_arr,
                pred_sem=p_sem,
                pred_bnd=p_bnd,
                pred_skl=p_skl,
                pred_instances=p_inst,
                metrics=metrics,
            )
            print(f"\n  [Visualizer] Saved 4-panel validation audit plot to: {vis_path}")

        print(
            f"  [Val Epoch {epoch:03d}] GT Inst: {total_gt_instances} (COCO: {total_coco_gt}) | Pred Inst: {total_pred_instances} | "
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
