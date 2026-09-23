from __future__ import annotations

import argparse
import os
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from dataset import SolarFilamentDataset, solar_collate_fn
from losses import MicroFilNetLoss
from model import MicroFilNet
from utils import (
    ModelEMA,
    binary_mask_to_rle,
    create_submission_csv,
    load_checkpoint,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MicroFilNet Full-Image CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_root", type=str, required=True)
    train_parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--batch_size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-5)
    train_parser.add_argument("--use_amp", action="store_true")
    train_parser.add_argument("--grad_clip", type=float, default=2.0)
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train_parser.add_argument("--val_plot_dir", type=str, default=None)
    train_parser.add_argument("--resume", type=str, default=None)
    train_parser.add_argument("--save_interval", type=int, default=5)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--use_ema", action="store_true")
    train_parser.add_argument("--ema_decay", type=float, default=0.9999)
    train_parser.add_argument("--val_split", type=float, default=0.1)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--weights", type=str, required=True)
    predict_parser.add_argument("--data_root", type=str, required=True)
    predict_parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--min_area", type=int, default=30)
    predict_parser.add_argument("--close_kernel", type=int, default=3)
    predict_parser.add_argument("--output", type=str, default="submission.csv")
    predict_parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    close_kernel_px: int = 3,
    min_area_px: int = 15,
) -> np.ndarray:
    binary = (prob_map >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    valid_labels = np.where((areas >= min_area_px) & (np.arange(n_labels) > 0))[0]
    return np.isin(labels, valid_labels).astype(np.uint8)


def extract_components_from_binary(binary: np.ndarray, min_area_px: int = 30) -> list[np.ndarray]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            component = (labels == lbl).astype(np.uint8)
            components.append(component)
    return components


def save_full_disk_validation_plot(
    model: torch.nn.Module,
    val_dataset: SolarFilamentDataset | Subset,
    device: torch.device,
    epoch: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    sample = val_dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    valid_mask = sample["valid_mask"].numpy()

    with torch.no_grad():
        preds = model(image)
        mask_logits = preds[0] if isinstance(preds, tuple) else preds
        probs = torch.sigmoid(mask_logits.float())[0, 0].cpu().numpy()

    probs = probs * valid_mask[0]
    pred_mask = (probs >= 0.5).astype(np.float32)
    gt_mask = sample["mask"][0].numpy() if sample["mask"] is not None else np.zeros_like(probs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(sample["image"][0].numpy(), cmap="gray")
    axes[0].set_title("H-alpha Full-Disk 2048x2048")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title(f"Ground Truth ({int(gt_mask.sum())} px)")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"Prediction ({int(pred_mask.sum())} px)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / f"val_full_disk_epoch_{epoch + 1}.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_training(args):
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_ddp:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_dir = Path(args.checkpoint_dir)
    val_plot_dir = Path(args.val_plot_dir) if args.val_plot_dir else (checkpoint_dir / "plots")
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        val_plot_dir.mkdir(parents=True, exist_ok=True)

    model = MicroFilNet().to(device)
    if use_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = MicroFilNetLoss()
    raw_model = model.module if use_ddp else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    ema = ModelEMA(raw_model, decay=args.ema_decay, device=device) if (args.use_ema and rank == 0) else None
    scaler = torch.amp.GradScaler("cuda") if (args.use_amp and device.type == "cuda") else None

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        chk_path = Path(args.resume)
        if chk_path == Path("last"):
            all_chk = list(checkpoint_dir.glob("checkpoint_*.pt"))
            chk_path = max(all_chk, key=os.path.getctime) if all_chk else (checkpoint_dir / "last.pt")

        if chk_path.exists():
            info = load_checkpoint(str(chk_path), raw_model, optimizer, ema, scheduler, str(device))
            start_epoch = info["epoch"] + 1
            best_loss = info["loss"]

    full_train_dataset = SolarFilamentDataset(
        data_root=args.data_root,
        split="train",
        use_cache=args.use_cache,
        augment=True,
    )

    full_val_dataset = SolarFilamentDataset(
        data_root=args.data_root,
        split="train",
        use_cache=args.use_cache,
        augment=False,
    )

    total_len = len(full_train_dataset)
    val_len = int(total_len * args.val_split)
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(total_len, generator=generator).tolist()
    train_indices = shuffled_indices[:train_len]
    val_indices = shuffled_indices[train_len:]

    train_ds = Subset(full_train_dataset, train_indices)
    val_ds = Subset(full_val_dataset, val_indices)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if (use_ddp and val_len > 0) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=solar_collate_fn,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=solar_collate_fn,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )
        if val_len > 0
        else None
    )

    for epoch in range(start_epoch, args.epochs):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        n_batches = len(train_loader)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False) if rank == 0 else train_loader
        for batch in iterator:
            images = batch["image"].to(device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(args.use_amp and device.type == "cuda")):
                preds = model(images)
                loss, _ = criterion(preds, masks, valid_masks, epoch)

            if args.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update(raw_model)

            loss_val = loss.item()
            total_loss += loss_val
            if rank == 0:
                iterator.set_postfix({"loss": f"{loss_val:.4f}"})

        current_loss = total_loss / max(n_batches, 1)

        if val_loader is not None:
            eval_target = ema.shadow_model if (ema is not None and rank == 0) else raw_model
            eval_target.eval()
            val_loss = 0.0
            val_batches = len(val_loader)

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device, non_blocking=True)
                    valid_masks = batch["valid_mask"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)

                    preds = eval_target(images)
                    l_val, _ = criterion(preds, masks, valid_masks, epoch)
                    val_loss += l_val.item()

            val_loss = val_loss / max(val_batches, 1)

            if use_ddp:
                loss_tensor = torch.tensor([val_loss], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                eval_loss = loss_tensor.item()
            else:
                eval_loss = val_loss

            if rank == 0:
                save_full_disk_validation_plot(
                    model=eval_target,
                    val_dataset=val_ds,
                    device=device,
                    epoch=epoch,
                    out_dir=val_plot_dir,
                )
        else:
            eval_loss = current_loss

        train_loss = total_loss / max(n_batches, 1)
        scheduler.step(eval_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        if rank == 0:
            if val_loader is not None:
                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] "
                    f"train_loss={train_loss:.4f} val_loss={eval_loss:.4f} lr={lr_now:.2e}",
                    flush=True,
                )
            else:
                print(
                    f"[Epoch {epoch + 1}/{args.epochs}] "
                    f"train_loss={train_loss:.4f} lr={lr_now:.2e}",
                    flush=True,
                )

            is_best = eval_loss < best_loss
            if is_best:
                best_loss = eval_loss

            save_checkpoint(
                model=raw_model,
                optimizer=optimizer,
                epoch=epoch,
                loss=eval_loss,
                filepath=str(checkpoint_dir / "last.pt"),
                ema_model=ema,
                scheduler=scheduler,
            )

            if is_best:
                save_checkpoint(
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=eval_loss,
                    filepath=str(checkpoint_dir / "best_model.pt"),
                    ema_model=ema,
                    scheduler=scheduler,
                )

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=eval_loss,
                    filepath=str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"),
                    ema_model=ema,
                    scheduler=scheduler,
                )

    if use_ddp:
        dist.destroy_process_group()


def run_prediction(args):
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = MicroFilNet().to(device)
    checkpoint_path = Path(args.weights)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Weights not found at {checkpoint_path}")

    load_checkpoint(str(checkpoint_path), model, device=str(device))
    model.eval()

    test_dataset = SolarFilamentDataset(
        data_root=args.data_root,
        split="test",
        use_cache=args.use_cache,
    )

    predictions = {}
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            image_id = sample["image_id"]
            image = sample["image"].unsqueeze(0).to(device)
            valid_mask = sample["valid_mask"].numpy()

            preds = model(image)
            mask_logits = preds[0] if isinstance(preds, tuple) else preds
            probs = torch.sigmoid(mask_logits.float())[0, 0].cpu().numpy()

            probs = probs * valid_mask[0]

            binary = postprocess_mask(
                probs,
                threshold=args.threshold,
                close_kernel_px=args.close_kernel,
                min_area_px=args.min_area,
            )

            components = extract_components_from_binary(binary, min_area_px=args.min_area)
            predictions[image_id] = [binary_mask_to_rle(c) for c in components]

    create_submission_csv(predictions, args.output)


def main():
    args = parse_args()
    if args.command == "train":
        run_training(args)
    elif args.command == "predict":
        run_prediction(args)


if __name__ == "__main__":
    main()