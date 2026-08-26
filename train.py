"""
train.py — Entry point for OPT-HQ Net training.

Usage
-----
    python train.py --data_root data/ --epochs 50 --batch_size 2

All hyper-parameters have sensible defaults from ``TrainingConfig``.
Override via CLI arguments.

Example (single GPU)
--------------------
    python train.py \
        --data_root /path/to/magfilo/ \
        --backbone convnext_large \
        --epochs 50 \
        --batch_size 2 \
        --lr 1e-4 \
        --device cuda

Example (quick smoke-test on CPU)
----------------------------------
    python train.py --data_root data/ --epochs 1 --device cpu --batch_size 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opt_hq_net.config import ModelConfig, TrainingConfig
from opt_hq_net.data import SolarFilamentDataset, collate_fn
from opt_hq_net.engine import Trainer
from opt_hq_net.models.opt_hq_net import OPTHQNetBuilder


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OPT-HQ Net on the MAGFiLO solar filament benchmark."
    )
    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing 'images/' and 'masks/' subdirs.")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Image size for training crops (default: 512).")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Crop patch size for foreground-centric training (e.g. 512 or 256).")
    parser.add_argument("--fg_patch_prob", type=float, default=0.8,
                        help="Probability of sampling patch centered on GT filament (default: 0.8).")
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        help="Backbone architecture (e.g. convnext_tiny, convnext_small, convnext_large, swin_large).")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet pretrained weights.")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of steps to accumulate gradients before optimizer step.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing for max speed if VRAM permits.")
    parser.add_argument("--use_multi_gpu", action="store_true", help="Enable nn.DataParallel across multiple GPUs.")
    parser.add_argument("--val_subset", type=int, default=0, help="Validate on first N images for rapid debugging (0 = full validation).")
    parser.add_argument("--val_every", type=int, default=1, help="Validation frequency in epochs (e.g. 5 = validate every 5 epochs).")
    parser.add_argument("--score_thresh", type=float, default=0.01, help="Minimum proposal score during validation.")

    # Loss weights
    parser.add_argument("--lambda_box",      type=float, default=1.0)
    parser.add_argument("--lambda_focal",    type=float, default=2.0)
    parser.add_argument("--lambda_dice",     type=float, default=2.0)
    parser.add_argument("--lambda_skeleton", type=float, default=1.5)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Worker for DistributedDataParallel (DDP)
# ---------------------------------------------------------------------------

def run_ddp_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    import os
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    # 1. Initialize DDP Process Group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2. Build configs
    model_cfg = ModelConfig(
        backbone_name=args.backbone,
        backbone_pretrained=not args.no_pretrained,
    )
    model_cfg.rpn.score_threshold = args.score_thresh

    train_cfg = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_amp=not args.no_amp,
        use_multi_gpu=True,
        checkpoint_dir=args.checkpoint_dir,
        device=f"cuda:{rank}",
        grad_checkpointing=not args.no_grad_ckpt,
        val_subset=args.val_subset,
        val_every_n_epochs=args.val_every,
    )
    train_cfg.loss_weights.oriented_box = args.lambda_box
    train_cfg.loss_weights.focal        = args.lambda_focal
    train_cfg.loss_weights.dice         = args.lambda_dice
    train_cfg.loss_weights.skeleton     = args.lambda_skeleton

    # 3. Build model
    if rank == 0:
        print(f"[DDP Rank 0] Building OPT-HQ Net with backbone: {model_cfg.backbone_name}")
    model = OPTHQNetBuilder(model_cfg).build()

    # 4. Build Datasets & DistributedSampler
    data_root_path = Path(args.data_root)
    train_path = data_root_path / "train" if (data_root_path / "train").exists() else data_root_path

    val_ds = None
    if (data_root_path / "val").exists():
        candidate_val = SolarFilamentDataset(
            data_root=data_root_path / "val",
            augment=False,
            target_size=args.target_size,
        )
        if candidate_val.has_masks:
            val_ds = candidate_val

    if val_ds is not None:
        train_ds = SolarFilamentDataset(
            data_root=train_path,
            augment=True,
            target_size=args.target_size,
            patch_size=args.patch_size,
            fg_patch_prob=args.fg_patch_prob,
        )
    else:
        full_train_ds = SolarFilamentDataset(
            data_root=train_path,
            augment=True,
            target_size=args.target_size,
        )
        full_val_ds = SolarFilamentDataset(
            data_root=train_path,
            augment=False,
            target_size=args.target_size,
        )
        val_size = max(1, int(0.2 * len(full_train_ds)))
        train_size = len(full_train_ds) - val_size
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_train_ds), generator=generator).tolist()
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_val_ds, val_indices)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=train_sampler,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(train_cfg.num_workers > 0),
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )

    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(train_cfg.num_workers > 0),
            prefetch_factor=2 if train_cfg.num_workers > 0 else None,
        )

    trainer = Trainer(model, train_loader, val_loader, train_cfg, rank=rank, world_size=world_size)
    trainer.train()

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Single Process / Single GPU execution
# ---------------------------------------------------------------------------

def run_single_process(args: argparse.Namespace) -> None:
    model_cfg = ModelConfig(
        backbone_name=args.backbone,
        backbone_pretrained=not args.no_pretrained,
    )
    model_cfg.rpn.score_threshold = args.score_thresh

    train_cfg = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_amp=not args.no_amp,
        use_multi_gpu=False,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        grad_checkpointing=not args.no_grad_ckpt,
        val_subset=args.val_subset,
        val_every_n_epochs=args.val_every,
    )

    train_cfg.loss_weights.oriented_box = args.lambda_box
    train_cfg.loss_weights.focal        = args.lambda_focal
    train_cfg.loss_weights.dice         = args.lambda_dice
    train_cfg.loss_weights.skeleton     = args.lambda_skeleton

    print(f"[Main] Building OPT-HQ Net with backbone: {model_cfg.backbone_name}")
    model = OPTHQNetBuilder(model_cfg).build()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Trainable parameters: {n_params / 1e6:.1f}M")

    data_root_path = Path(args.data_root)
    train_path = data_root_path / "train" if (data_root_path / "train").exists() else data_root_path

    print(f"[Main] Loading train dataset from: {train_path}")

    val_ds = None
    if (data_root_path / "val").exists():
        candidate_val = SolarFilamentDataset(
            data_root=data_root_path / "val",
            augment=False,
            target_size=args.target_size,
        )
        if candidate_val.has_masks:
            val_ds = candidate_val
            print(f"[Main] Loading val dataset from:   {data_root_path / 'val'}")

    if val_ds is not None:
        train_ds = SolarFilamentDataset(
            data_root=train_path,
            augment=True,
            target_size=args.target_size,
            patch_size=args.patch_size,
            fg_patch_prob=args.fg_patch_prob,
        )
    else:
        full_train_ds = SolarFilamentDataset(
            data_root=train_path,
            augment=True,
            target_size=args.target_size,
        )
        full_val_ds = SolarFilamentDataset(
            data_root=train_path,
            augment=False,
            target_size=args.target_size,
        )
        val_size = max(1, int(0.2 * len(full_train_ds)))
        train_size = len(full_train_ds) - val_size

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_train_ds), generator=generator).tolist()
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        train_ds = torch.utils.data.Subset(full_train_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_val_ds, val_indices)
        print(f"[Main] Auto Validation Split created: {train_size} train images | {val_size} val images (80/20 split)")

    train_loader_kwargs = {
        "batch_size": train_cfg.batch_size,
        "shuffle": True,
        "num_workers": train_cfg.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": (args.device == "cuda"),
    }
    val_loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": train_cfg.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": (args.device == "cuda"),
    }

    if train_cfg.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    print(f"[Main] Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    trainer = Trainer(model, train_loader, val_loader, train_cfg)
    trainer.train()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.use_multi_gpu and num_gpus > 1:
        import torch.multiprocessing as mp
        print(f"[Main] Launching DistributedDataParallel (DDP) across {num_gpus} GPUs...")
        mp.spawn(run_ddp_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
    else:
        run_single_process(args)


if __name__ == "__main__":
    main()
