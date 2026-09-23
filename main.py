from __future__ import annotations

import argparse
import os
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe: training runs on servers with no display
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
from inference import postprocess_mask, tiled_predict
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
    parser = argparse.ArgumentParser(description="MicroFilNet Main CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_root", type=str, required=True)
    train_parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--tile_size", type=int, default=512)
    train_parser.add_argument("--overlap", type=float, default=0.25)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-5)
    train_parser.add_argument("--use_amp", action="store_true")
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
    predict_parser.add_argument("--tile_size", type=int, default=512)
    predict_parser.add_argument("--overlap", type=float, default=0.25)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--min_area", type=int, default=30)
    predict_parser.add_argument("--close_kernel", type=int, default=3)
    predict_parser.add_argument("--output", type=str, default="submission.csv")
    predict_parser.add_argument("--device", type=str, default="cuda")
    predict_parser.add_argument("--batch_size", type=int, default=4)

    return parser.parse_args()


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
    val_subset: Subset,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    tile_size: int = 512,
    overlap: float = 0.25,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    underlying_dataset: SolarFilamentDataset = val_subset.dataset
    chosen_idx = val_subset.indices[0]
    for idx in val_subset.indices:
        file_name = underlying_dataset.image_files[idx].name
        gt_mask = underlying_dataset._generate_mask(file_name, fallback_shape=(2048, 2048))
        if gt_mask is not None and gt_mask.sum() > 50:
            chosen_idx = idx
            break

    raw_path = underlying_dataset.image_files[chosen_idx]
    clean_img, valid_mask, (cx, cy, r_sun) = underlying_dataset._get_processed_data(raw_path)
    h, w = clean_img.shape

    image_stack = np.expand_dims(clean_img, axis=0)
    valid_mask_stack = np.expand_dims(valid_mask, axis=0)

    global_img = cv2.resize(clean_img, (512, 512), interpolation=cv2.INTER_AREA)
    global_stack = np.expand_dims(global_img, axis=0).astype(np.float32)

    prob_map = tiled_predict(
        model=model,
        image=image_stack,
        valid_mask=valid_mask_stack,
        global_image=global_stack,
        disk_center=(cx, cy, r_sun),
        tile=tile_size,
        overlap=overlap,
        device=device,
        batch_size=4,
    )

    pred_mask = (prob_map >= 0.5).astype(np.float32)
    gt_mask = underlying_dataset._generate_mask(raw_path.name, fallback_shape=(h, w))
    if gt_mask is None:
        gt_mask = np.zeros((h, w), dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(clean_img, cmap="gray")
    axes[0].set_title(f"H-alpha Normalized ({h}x{w})")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_cache=args.use_cache,
        augment=True,
    )

    full_val_dataset = SolarFilamentDataset(
        data_root=args.data_root,
        split="train",
        tile_size=args.tile_size,
        overlap=args.overlap,
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
            global_images = batch["global_image"].to(device, non_blocking=True)
            coords = batch["coords"].to(device, non_blocking=True)
            valid_masks = batch["valid_mask"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(args.use_amp and device.type == "cuda")):
                logits = model(images, global_images, coords)

            loss, _ = criterion(logits.float(), masks.float(), valid_masks.float(), epoch)

            if args.use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
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
                    global_images = batch["global_image"].to(device, non_blocking=True)
                    coords = batch["coords"].to(device, non_blocking=True)
                    valid_masks = batch["valid_mask"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    logits = eval_target(images, global_images, coords)
                    l_val, _ = criterion(logits.float(), masks.float(), valid_masks.float(), epoch)
                    val_loss += l_val.item()

            val_loss = val_loss / max(val_batches, 1)

            if use_ddp:
                loss_tensor = torch.tensor([val_loss], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                current_loss = loss_tensor.item()
            else:
                current_loss = val_loss

            if rank == 0:
                save_full_disk_validation_plot(
                    model=eval_target,
                    val_subset=val_ds,
                    device=device,
                    epoch=epoch,
                    out_dir=val_plot_dir,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                )

        scheduler.step()

        if rank == 0:
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss

            save_checkpoint(
                model=raw_model,
                optimizer=optimizer,
                epoch=epoch,
                loss=current_loss,
                filepath=str(checkpoint_dir / "last.pt"),
                ema_model=ema,
                scheduler=scheduler,
            )

            if is_best:
                save_checkpoint(
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=current_loss,
                    filepath=str(checkpoint_dir / "best_model.pt"),
                    ema_model=ema,
                    scheduler=scheduler,
                )

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=current_loss,
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
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_cache=args.use_cache,
    )

    predictions = {}
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            image_id = sample["image_id"]
            image = sample["image"].numpy()
            global_image = sample["global_image"].numpy()
            valid_mask = sample["valid_mask"].numpy()
            disk = sample["disk"]

            prob_map = tiled_predict(
                model=model,
                image=image,
                valid_mask=valid_mask,
                global_image=global_image,
                disk_center=disk,
                tile=args.tile_size,
                overlap=args.overlap,
                device=device,
                batch_size=args.batch_size,
            )

            binary = postprocess_mask(
                prob_map,
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