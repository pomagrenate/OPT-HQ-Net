from __future__ import annotations

import argparse
from pathlib import Path
import torch

from dataset import SolarFilamentDataset
from inference import tiled_predict
from model import MicroFilNet
from predict import postprocess_and_extract_components
from train import evaluate, train_one_epoch
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
    train_parser.add_argument("--use_cache", action="store_true", default=True)
    train_parser.add_argument("--tile_size", type=int, default=256)
    train_parser.add_argument("--overlap", type=float, default=0.25)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-5)
    train_parser.add_argument("--use_amp", action="store_true")
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train_parser.add_argument("--resume", type=str, default=None)
    train_parser.add_argument("--save_interval", type=int, default=5)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--use_ema", action="store_true")
    train_parser.add_argument("--ema_decay", type=float, default=0.9999)
    train_parser.add_argument("--val_split", type=float, default=0.1)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--weights", type=str, required=True)
    predict_parser.add_argument("--data_root", type=str, required=True)
    predict_parser.add_argument("--use_cache", action="store_true", default=True)
    predict_parser.add_argument("--tile_size", type=int, default=256)
    predict_parser.add_argument("--overlap", type=float, default=0.25)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--min_area", type=int, default=30)
    predict_parser.add_argument("--close_kernel", type=int, default=3)
    predict_parser.add_argument("--output", type=str, default="submission.csv")
    predict_parser.add_argument("--device", type=str, default="cuda")
    predict_parser.add_argument("--batch_size", type=int, default=8)

    return parser.parse_args()


def run_training(args):
    from torch.utils.data import DataLoader
    from dataset import solar_collate_fn
    from losses import MicroFilNetLoss

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = MicroFilNet().to(device)
    criterion = MicroFilNetLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.use_ema else None
    scaler = torch.amp.GradScaler("cuda") if (args.use_amp and device.type == "cuda") else None

    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        chk_path = Path(args.resume)
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
            eval_model = ema.shadow_model if ema is not None else model
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


def run_prediction(args):
    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

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
            valid_mask = sample["valid_mask"].numpy()

            prob_map = tiled_predict(
                model,
                image,
                valid_mask,
                tile=args.tile_size,
                overlap=args.overlap,
                device=device,
                batch_size=args.batch_size,
            )

            components = postprocess_and_extract_components(
                prob_map,
                threshold=args.threshold,
                close_kernel_px=args.close_kernel,
                min_area_px=args.min_area,
            )

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