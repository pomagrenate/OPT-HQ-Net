"""
Production-Ready, High-Throughput Trainer Engine for Solar Filament Micro-Segmentation.

Features:
  - Multi-GPU DistributedDataParallel (DDP) across Kaggle 2x T4/P100 and clean single-GPU execution.
  - Automatic Mixed Precision (AMP FP16) with GradScaler.
  - Full Checkpoint Resumption (recovering weights, optimizer, scaler, EMA, epoch, and metrics).
  - Emergency interrupt safety on KeyboardInterrupt / SIGINT.
  - Model EMA shadow parameter tracking.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.losses import CompoundLoss
from src.utils import ModelEMA, PanopticQualityMetric


_MAX_DEFAULT_VAL_BATCHES = 500  # Safety cap: prevents >10min val loops causing NCCL watchdog timeouts


class SolarTrainer:
    """
    Unified High-Performance Trainer Engine.

    Parameters
    ----------
    model : nn.Module
        SolarFilamentNet instance.
    train_loader : DataLoader
        PyTorch DataLoader for training patches.
    val_loader : Optional[DataLoader]
        Optional DataLoader for validation patches.
    optimizer : Optional[torch.optim.Optimizer]
        AdamW optimizer instance.
    epochs : int
        Target total epochs (default: 50).
    device : str | torch.device
        Target compute device.
    use_amp : bool
        Enable Automatic Mixed Precision FP16 (default: True).
    checkpoint_dir : str | Path
        Directory to save and load checkpoints (default: 'checkpoints').
    resume : Optional[str | Path]
        Path to checkpoint to resume training from, or 'last'.
    save_interval : int
        Epoch interval for saving milestone checkpoints (default: 1).
    rank : int
        Process rank for DDP (default: 0).
    world_size : int
        Total process count for DDP (default: 1).
    max_val_batches : int
        Maximum number of validation batches per epoch to prevent NCCL watchdog timeouts.
        Default 500 (~500 * batch_size=4 = 2000 tiles, fast enough to not stall DDP ranks).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 50,
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        checkpoint_dir: str | Path = "checkpoints",
        resume: Optional[Union[str, Path, bool]] = None,
        save_interval: int = 1,
        rank: int = 0,
        world_size: int = 1,
        max_val_batches: int = _MAX_DEFAULT_VAL_BATCHES,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.is_master = (rank == 0)

        # Device setup
        if world_size > 1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.save_interval = max(1, save_interval)
        self.max_val_batches = max(10, max_val_batches)  # Safety floor: always run ≥10 val batches
        self.checkpoint_dir = Path(checkpoint_dir)
        if self.is_master:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimization & Memory Flags
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.raw_model = model.to(self.device)

        # Multi-GPU Wrapping
        if world_size > 1:
            self.model = DDP(self.raw_model, device_ids=[rank], output_device=rank)
            if self.is_master:
                print(f"[SolarTrainer] DDP Initialized across {world_size} GPUs (Local Rank {rank})")
        elif torch.cuda.device_count() > 1 and not (world_size > 1):
            if self.is_master:
                print(f"[SolarTrainer] Enabling DataParallel across {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.raw_model)
        else:
            self.model = self.raw_model
            if self.is_master:
                print(f"[SolarTrainer] Single GPU/CPU execution on {self.device}")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer or torch.optim.AdamW(
            self.raw_model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.loss_fn = CompoundLoss().to(self.device)

        # Model EMA (evaluated on master)
        self.ema = ModelEMA(self.raw_model, decay=0.999, device=str(self.device)) if self.is_master else None

        # AMP Scaler
        self.use_amp = use_amp and (self.device.type == "cuda")
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.metric = PanopticQualityMetric(iou_threshold=0.5)

        self.start_epoch = 1
        self.best_dice = 0.0

        if resume:
            self.load_checkpoint(resume)

    def resolve_checkpoint_path(self, resume: Union[str, Path, bool]) -> Path:
        """Resolve checkpoint path from file path, keyword ('last', 'latest'), or boolean."""
        if isinstance(resume, bool) or str(resume).lower() in ("last", "latest", "true"):
            last_path = self.checkpoint_dir / "last.pt"
            if last_path.is_file():
                return last_path

            interrupted_path = self.checkpoint_dir / "checkpoint_interrupted.pt"
            if interrupted_path.is_file():
                return interrupted_path

            epoch_ckpts = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if epoch_ckpts:
                return epoch_ckpts[-1]

            best_path = self.checkpoint_dir / "best_model.pt"
            if best_path.is_file():
                return best_path

            raise FileNotFoundError(f"No valid checkpoint found in '{self.checkpoint_dir}' to resume from.")

        p = Path(resume)
        if p.is_file():
            return p
        in_dir = self.checkpoint_dir / resume
        if in_dir.is_file():
            return in_dir

        raise FileNotFoundError(f"Specified checkpoint not found: '{resume}'")

    def load_checkpoint(self, checkpoint_path: Union[str, Path, bool]) -> None:
        """Load model weights, optimizer moments, scaler state, and epoch counters."""
        path = self.resolve_checkpoint_path(checkpoint_path)
        if self.is_master:
            print(f"\n[SolarTrainer] Resuming training from checkpoint: {path}")

        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.raw_model.load_state_dict(ckpt["model_state_dict"])
            if self.is_master:
                print("  [OK] Model weights restored.")

            if "optimizer_state_dict" in ckpt and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if self.is_master:
                        print("  [OK] Optimizer state restored.")
                except Exception as err:
                    if self.is_master:
                        print(f"  [WARN] Could not restore optimizer state ({err}). Starting optimizer fresh.")

            if self.use_amp and self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
                try:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
                    if self.is_master:
                        print("  [OK] AMP GradScaler restored.")
                except Exception as err:
                    if self.is_master:
                        print(f"  [WARN] Could not restore scaler state ({err}).")

            if self.ema is not None and ckpt.get("ema_state_dict") is not None:
                try:
                    self.ema.module.load_state_dict(ckpt["ema_state_dict"])
                    if self.is_master:
                        print("  [OK] Model EMA weights restored.")
                except Exception:
                    self.ema.set(self.raw_model)
            elif self.ema is not None:
                self.ema.set(self.raw_model)

            last_epoch = ckpt.get("epoch", 0)
            self.start_epoch = last_epoch + 1
            self.best_dice = float(ckpt.get("best_dice", 0.0))

            if self.is_master:
                print(
                    f"  [OK] Resuming forward from Epoch {self.start_epoch:03d} "
                    f"(Completed: {last_epoch}, Historical Best Dice: {self.best_dice:.4f})\n"
                )
        elif isinstance(ckpt, dict):
            self.raw_model.load_state_dict(ckpt)
            if self.ema is not None:
                self.ema.set(self.raw_model)
            if self.is_master:
                print("  [INFO] Loaded raw model state_dict. Starting from Epoch 1 with fresh optimizer.\n")

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_interrupted: bool = False,
    ) -> None:
        """Save full training state to checkpoint directory."""
        if not self.is_master:
            return

        state = {
            "epoch": epoch,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "ema_state_dict": self.ema.module.state_dict() if self.ema is not None else None,
            "best_dice": self.best_dice,
        }

        # 1. Always update last.pt on completed epoch or interrupt
        torch.save(state, self.checkpoint_dir / "last.pt")

        if is_interrupted:
            int_path = self.checkpoint_dir / "checkpoint_interrupted.pt"
            torch.save(state, int_path)
            print(f"\n  [Checkpoint] Saved emergency interrupt checkpoint to: {int_path}")
            return

        # 2. Save periodic milestone checkpoint
        if (epoch % self.save_interval == 0) or epoch == self.epochs:
            torch.save(state, self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        # 3. Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            print(f"  [Checkpoint] Saved new BEST model (Dice: {self.best_dice:.4f}) to {best_path}")

    def train(self) -> Dict[str, float]:
        """Run the main multi-epoch training loop."""
        if self.is_master:
            print(f"\n[SolarTrainer] Starting Training on {self.device}")
            print(f"  AMP: {self.use_amp} | Start Epoch: {self.start_epoch} | Target Epochs: {self.epochs}")
            print(f"  Train Batches: {len(self.train_loader)} | Checkpoint Dir: {self.checkpoint_dir}\n")

        if self.start_epoch > self.epochs:
            if self.is_master:
                print(f"[SolarTrainer] Target epochs ({self.epochs}) already achieved. Nothing to train.")
            return {"best_dice": self.best_dice}

        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                # Set epoch on DistributedSampler if active
                if hasattr(self.train_loader, "sampler") and isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

                train_metrics = self._train_one_epoch(epoch)

                if self.is_master:
                    print(
                        f"[Epoch {epoch:04d}/{self.epochs:04d}] "
                        f"Loss: {train_metrics['loss']:.4f} | "
                        f"BCE: {train_metrics['loss_bce']:.4f} | "
                        f"Dice: {train_metrics['loss_dice']:.4f} | "
                        f"clDice: {train_metrics['loss_cldice']:.4f} | "
                        f"Skel: {train_metrics['loss_skel']:.4f}"
                    )

                # Validation — must be wrapped in DDP barriers to prevent NCCL watchdog timeout.
                # Root cause of prior crash: Rank 0 runs long val loop while Rank 1 sits idle.
                # The NCCL watchdog on Rank 1 fires after 600s, triggering SIGABRT on both ranks.
                # Fix: ALL ranks enter a barrier BEFORE val starts, then again AFTER val completes.
                is_best = False
                do_validate = (self.val_loader is not None and (epoch % 2 == 0 or epoch == self.epochs))

                # All ranks synchronize before validation gate
                if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                    dist.barrier()

                if do_validate and self.is_master:
                    val_metrics = self._validate(epoch)
                    curr_dice = val_metrics.get("mean_dice", 0.0)
                    if curr_dice > self.best_dice:
                        self.best_dice = curr_dice
                        is_best = True

                # All ranks synchronize again after validation completes
                if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                    dist.barrier()

                self._save_checkpoint(epoch, is_best=is_best)

        except KeyboardInterrupt:
            if self.is_master:
                print("\n\n" + "=" * 65)
                print("  [WARNING] TRAINING INTERRUPTED BY USER!")
                interrupted_epoch = max(epoch - 1, 0)
                self._save_checkpoint(interrupted_epoch, is_interrupted=True)
                print(f"To continue training, run:")
                print(f"  python train.py --resume {self.checkpoint_dir / 'last.pt'}")
                print("=" * 65 + "\n")
            return {"best_dice": self.best_dice, "interrupted": True}

        return {"best_dice": self.best_dice}

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses: Dict[str, float] = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train]", disable=not self.is_master, leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            # Skeleton target is available but not used in training (GPU soft-skeleton is used in loss)
            # skeletons = batch.get("skeleton").to(self.device, non_blocking=True) if "skeleton" in batch else None

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                try:
                    autocast_cm = torch.amp.autocast("cuda")
                except AttributeError:
                    autocast_cm = torch.cuda.amp.autocast()

                with autocast_cm:
                    preds = self.model(images)
                    loss_dict = self.loss_fn(preds, masks)

                self.scaler.scale(loss_dict["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                loss_dict = self.loss_fn(preds, masks)
                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.raw_model)

            # Accumulate loss metrics
            for k, v in loss_dict.items():
                val_float = v.item() if isinstance(v, torch.Tensor) else float(v)
                running_losses[k] = running_losses.get(k, 0.0) + val_float

            pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

        num_batches = max(len(self.train_loader), 1)
        return {k: v / num_batches for k, v in running_losses.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        eval_model = self.ema.module if self.ema is not None else self.raw_model
        eval_model.eval()
        self.metric.reset()

        total_val = len(self.val_loader)
        capped = min(total_val, self.max_val_batches)
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:03d} [Val  ]", total=capped, leave=False)

        for step, batch in enumerate(pbar):
            if step >= self.max_val_batches:
                break

            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].cpu().numpy()

            if self.use_amp:
                try:
                    autocast_cm = torch.amp.autocast("cuda")
                except AttributeError:
                    autocast_cm = torch.cuda.amp.autocast()
                with autocast_cm:
                    logits = eval_model(images)
            else:
                logits = eval_model(images)

            mask_probs = torch.sigmoid(logits[:, 0:1]).cpu().numpy()

            for b in range(images.shape[0]):
                p_bin = (mask_probs[b, 0] > 0.50).astype(np.uint8)
                g_bin = (masks[b, 0] > 0.50).astype(np.uint8)
                self.metric.update(p_bin, g_bin)

        metrics = self.metric.compute()
        print(
            f"  [Val Epoch {epoch:03d}] Mean Dice: {metrics['mean_dice']:.4f} | "
            f"PQ: {metrics['PQ']:.4f} | TP: {metrics['TP']} | FP: {metrics['FP']} | FN: {metrics['FN']}"
            f" | Batches: {min(capped, total_val)}/{total_val}"
        )
        return metrics
