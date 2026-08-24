"""
Training engine for OPT-HQ Net.

Features
--------
- AdamW optimiser with Cosine Annealing LR schedule and linear warmup.
- Automatic Mixed Precision (AMP/FP16) via ``torch.amp.GradScaler``.
- Gradient clipping to prevent exploding gradients during early epochs.
- Periodic checkpoint saving.
- Per-epoch metric evaluation using ``PanopticQualityMetric``.
- Rich progress bar display via ``tqdm``.

Usage
-----
>>> from opt_hq_net import ModelConfig, TrainingConfig, OPTHQNetBuilder
>>> from opt_hq_net.engine import Trainer
>>> from opt_hq_net.data import SolarFilamentDataset, collate_fn
>>> from torch.utils.data import DataLoader
>>>
>>> model_cfg = ModelConfig()
>>> train_cfg = TrainingConfig()
>>> model = OPTHQNetBuilder(model_cfg).build()
>>>
>>> train_ds = SolarFilamentDataset("data/train", augment=True)
>>> val_ds   = SolarFilamentDataset("data/val",   augment=False)
>>> train_loader = DataLoader(train_ds, batch_size=2, collate_fn=collate_fn)
>>> val_loader   = DataLoader(val_ds,   batch_size=1, collate_fn=collate_fn)
>>>
>>> trainer = Trainer(model, train_loader, val_loader, train_cfg)
>>> trainer.train()
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from opt_hq_net.config import TrainingConfig
from opt_hq_net.metrics.panoptic_quality import PanopticQualityMetric


class Trainer:
    """
    Full training loop for OPT-HQ Net.

    Parameters
    ----------
    model : nn.Module
        OPTHQNet instance.
    train_loader : DataLoader
        Training data loader (uses ``collate_fn`` from ``opt_hq_net.data``).
    val_loader : DataLoader | None
        Validation data loader.  If None, validation is skipped.
    cfg : TrainingConfig
        Training hyper-parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: TrainingConfig,
    ) -> None:
        # Optimise CUDA memory allocation to prevent fragmentation OOM
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        # Device validation with sm_60 compatibility check
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            try:
                dummy = torch.zeros((1, 1), device=cfg.device)
                dummy = dummy + 1.0
                self.device = torch.device(cfg.device)
            except Exception as err:
                print(f"\n[Trainer WARNING] CUDA execution test failed: {err}")
                print("[Trainer HINT] Tesla P100 (sm_60) is not supported by PyTorch 2.4+ builds.")
                print("              In Kaggle Notebook settings -> 'Accelerator', change GPU from 'P100' to 'GPU T4 x2' (sm_75).\n")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Multi-GPU support (DataParallel) — disabled by default to avoid list target scattering bugs
        self.num_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if self.num_gpus > 1 and getattr(cfg, "use_multi_gpu", False):
            print(f"[Trainer] Multi-GPU setup detected: {self.num_gpus} GPUs available. Enabling nn.DataParallel!")
            self.model = nn.DataParallel(self.model)
            self.is_multi_gpu = True
        else:
            print(f"[Trainer] Single GPU execution on device: {self.device} (VRAM efficient mode)")
            self.is_multi_gpu = False

        # Configure gradient checkpointing on backbone
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(raw_model, "backbone") and hasattr(raw_model.backbone, "set_grad_checkpointing"):
            enable_gc = getattr(cfg, "grad_checkpointing", True)
            raw_model.backbone.set_grad_checkpointing(enable_gc)

        # Optimiser
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.optimizer = AdamW(
            raw_model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler with linear warmup + cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(cfg.num_epochs - cfg.warmup_epochs, 1),
            eta_min=cfg.learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.warmup_epochs],
        )

        # AMP
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
            except TypeError:
                self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            from torch.cuda.amp import GradScaler as LegacyGradScaler
            self.scaler = LegacyGradScaler(enabled=self.use_amp)

        # Metrics
        self.metric = PanopticQualityMetric(iou_threshold=0.5)

        # Checkpoint directory
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._best_pq = 0.0

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full training loop for ``cfg.num_epochs`` epochs."""
        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"[Trainer] Starting training on device: {self.device}")
        print(f"[Trainer] AMP enabled: {self.use_amp}")
        print(f"[Trainer] Multi-GPU active: {self.is_multi_gpu} ({self.num_gpus} GPUs detected)")
        print(f"[Trainer] Epochs: {self.cfg.num_epochs}")
        print(f"[Trainer] Batch size: {self.cfg.batch_size} (Grad Accum Steps: {getattr(self.cfg, 'gradient_accumulation_steps', 1)})")

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_losses = self._train_one_epoch(epoch)
            self.scheduler.step()

            # Validation
            val_every = getattr(self.cfg, "val_every_n_epochs", 1)
            if self.val_loader is not None and (epoch % val_every == 0):
                val_metrics = self._validate(epoch)
                pq = val_metrics.get("PQ", 0.0)
                if pq > self._best_pq:
                    self._best_pq = pq
                    self._save_checkpoint(epoch, tag="best")
                    print(f"  ✓ New best PQ: {pq:.4f}")
            else:
                val_metrics = {}

            # Periodic checkpoint
            if epoch % self.cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

            self._log_epoch(epoch, train_losses, val_metrics)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"[Trainer] Training complete. Best PQ: {self._best_pq:.4f}")

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch with gradient accumulation. Returns averaged loss values."""
        self.model.train()
        total_losses: Dict[str, float] = {}
        num_batches = 0
        grad_accum = getattr(self.cfg, "gradient_accumulation_steps", 1)

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d}/{self.cfg.num_epochs:03d} [Train]",
            leave=False,
            disable=not has_tqdm,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            images = batch["images"].to(self.device)
            gt_boxes = [b.to(self.device) for b in batch["boxes"]]
            gt_masks = [m.to(self.device) for m in batch["masks"]]

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(self.device.type, enabled=self.use_amp)
            else:
                from torch.cuda.amp import autocast as legacy_autocast
                autocast_ctx = legacy_autocast(enabled=self.use_amp)

            with autocast_ctx:
                _, loss_dict = self.model(images, gt_boxes=gt_boxes, gt_masks=gt_masks)

            if not loss_dict:
                continue

            # Handle multi-GPU loss vector gathering
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    loss_dict[k] = v.mean()

            total_loss = loss_dict["total_loss"] / grad_accum

            # Backward pass (scaled for accumulation)
            self.scaler.scale(total_loss).backward()

            # Optimizer step every grad_accum steps or at epoch end
            if (step + 1) % grad_accum == 0 or (step + 1) == len(self.train_loader):
                raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Accumulate unscaled losses for logging
            for k, v in loss_dict.items():
                unscaled_val = v.item() if k != "total_loss" else (total_loss.item() * grad_accum)
                total_losses[k] = total_losses.get(k, 0.0) + unscaled_val
            num_batches += 1

            if has_tqdm:
                pbar.set_postfix({"loss": f"{(total_loss.item() * grad_accum):.4f}"})

        if num_batches == 0:
            return {"total_loss": 0.0}

        return {k: v / num_batches for k, v in total_losses.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Validation loop — computes PQ on the validation set with verbose diagnostics."""
        self.model.eval()
        self.metric.reset()

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        val_subset = getattr(self.cfg, "val_subset", 0)

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d}/{self.cfg.num_epochs:03d} [Val  ]",
            leave=False,
            disable=not has_tqdm,
        )

        total_gt_instances = 0
        total_pred_instances = 0
        total_gt_pixels = 0
        total_pred_pixels = 0

        for step, batch in enumerate(pbar):
            if val_subset > 0 and step >= val_subset:
                break

            images = batch["images"].to(self.device)
            gt_masks_list = batch["masks"]   # list of Tensors

            predictions, _ = self.model(images)

            # Convert to numpy for metric computation
            pred_np = []
            for p in predictions:
                masks = p["masks"]
                if len(masks) > 0:
                    bin_masks = (masks > 0.5).cpu().numpy().astype("uint8")
                    pred_np.append(bin_masks)
                    total_pred_instances += len(bin_masks)
                    total_pred_pixels += int(bin_masks.sum())
                else:
                    pred_np.append(__import__("numpy").zeros((0, 1, 1), dtype="uint8"))

            gt_np = []
            for m in gt_masks_list:
                m_np = m.cpu().numpy().astype("uint8")
                gt_np.append(m_np)
                total_gt_instances += len(m_np)
                total_gt_pixels += int(m_np.sum())

            self.metric.update(pred_np, gt_np)

            # Diagnostic log for first few validation batches
            if step < 3 or (step + 1) == len(self.val_loader):
                batch_pred = sum(len(p) for p in pred_np)
                batch_gt = sum(len(g) for g in gt_np)
                scores_list = [p["scores"] for p in predictions if len(p["scores"]) > 0]
                max_score = float(torch.cat(scores_list).max()) if scores_list and len(torch.cat(scores_list)) > 0 else 0.0
                print(
                    f"\n  [Val Debug Step {step+1:03d}] GT Instances: {batch_gt} | "
                    f"Pred Instances: {batch_pred} (Max Score: {max_score:.4f}) | "
                    f"Pred FG Pixels: {sum(p.sum() for p in pred_np)} | GT FG Pixels: {sum(g.sum() for g in gt_np)}"
                )

        metrics = self.metric.compute()
        print(
            f"  [Val Overview] GT Instances: {total_gt_instances} | "
            f"Pred Instances: {total_pred_instances} | "
            f"TP: {metrics.get('TP', 0)} | FP: {metrics.get('FP', 0)} | FN: {metrics.get('FN', 0)} | "
            f"Mean Dice: {metrics.get('mean_dice', 0.0):.4f} | PQ: {metrics.get('PQ', 0.0):.4f}"
        )

        return metrics

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, tag: str = "epoch") -> None:
        """Save model weights and optimiser state."""
        path = self.ckpt_dir / f"opt_hq_net_{tag}_{epoch:04d}.pth"
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_pq": self._best_pq,
        }
        torch.save(ckpt_data, str(path))

        # Also save static best model file for easy loading in predict.py
        if tag == "best":
            best_static_path = self.ckpt_dir / "opt_hq_net_best.pth"
            torch.save(ckpt_data, str(best_static_path))
            print(f"  Saved best model checkpoint to: {best_static_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _log_epoch(epoch: int, train_losses: Dict, val_metrics: Dict) -> None:
        loss_str = "  ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
        val_str = "  ".join(f"{k}: {v}" for k, v in val_metrics.items())
        print(f"[Epoch {epoch:04d}] TRAIN → {loss_str}")
        if val_str:
            print(f"           VAL   → {val_str}")

    # ------------------------------------------------------------------
    @classmethod
    def load_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> nn.Module:
        """
        Load model weights from a checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        state_dict = ckpt["model_state_dict"]

        # Strip module. prefix if needed
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[key] = v

        raw_model.load_state_dict(cleaned_state_dict)
        print(f"[Trainer] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
              f"best PQ: {ckpt.get('best_pq', 0.0):.4f}")
        return model
