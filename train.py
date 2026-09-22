from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SolarFilamentDataset, solar_collate_fn
from losses import MicroFilNetLoss
from model import MicroFilNet
from utils import ModelEMA, load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train MicroFilNet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--val_split", type=float, default=0.1)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    use_amp: bool,
    ema: Optional[ModelEMA] = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    accum_parts = {"bce": 0.0, "dice": 0.0, "cldice": 0.0, "boundary": 0.0}
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        valid_masks = batch["valid_mask"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, parts = criterion(logits, masks, valid_masks, epoch)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_val = loss.item()
        total_loss += loss_val
        for k in accum_parts:
            accum_parts[k] += parts[k].item()

        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    metrics = {k: v / max(n_batches, 1) for k, v in accum_parts.items()}
    metrics["total_loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    accum_parts = {"bce": 0.0, "dice": 0.0, "cldice": 0.0, "boundary": 0.0}
    n_batches = len(dataloader)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        valid_masks = batch["valid_mask"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss, parts = criterion(logits, masks, valid_masks, epoch)

        total_loss += loss.item()
        for k in accum_parts:
            accum_parts[k] += parts[k].item()

    metrics = {k: v / max(n_batches, 1) for k, v in accum_parts.items()}
    metrics["total_loss"] = total_loss / max(n_batches, 1)
    return metrics


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = MicroFilNet().to(device)
    criterion = MicroFilNetLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.use_ema else None
    scaler = torch.amp.GradScaler("cuda") if (args.use_amp and device.type == "cuda") else None

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        chk_path = Path(args.resume)
        if chk_path == Path("last"):
            all_chk = list(checkpoint_dir.glob("checkpoint_*.pt"))
            chk_path = max(all_chk, key=os.path.getctime) if all_chk else (checkpoint_dir / "last.pt")

        if chk_path.exists():
            info = load_checkpoint(str(chk_path), model, optimizer, ema, scheduler, str(device))
            start_epoch = info["epoch"] + 1
            best_loss = info["loss"]

    full_dataset = SolarFilamentDataset(
        data_root=args.data_root,
        split="train",
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_cache=args.use_cache,
        augment=True,
    )

    total_len = len(full_dataset)
    val_len = int(total_len * args.val_split)
    train_len = total_len - val_len

    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=(args.use_amp and device.type == "cuda"),
            ema=ema,
        )

        current_loss = train_metrics["total_loss"]
        if val_loader is not None:
            eval_model = ema.shadow_model if (ema is not None and hasattr(ema, "shadow_model")) else model
            val_metrics = evaluate(eval_model, val_loader, criterion, device, epoch)
            current_loss = val_metrics["total_loss"]

        scheduler.step()

        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=current_loss,
            filepath=str(checkpoint_dir / "last.pt"),
            ema_model=ema,
            scheduler=scheduler,
        )

        if is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=current_loss,
                filepath=str(checkpoint_dir / "best_model.pt"),
                ema_model=ema,
                scheduler=scheduler,
            )

        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=current_loss,
                filepath=str(checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"),
                ema_model=ema,
                scheduler=scheduler,
            )


if __name__ == "__main__":
    main()