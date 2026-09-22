from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import torch
import torch.nn as nn

from dataset import SolarFilamentDataset
from inference import tiled_predict
from model import MicroFilNet
from utils import binary_mask_to_rle, create_submission_csv, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with MicroFilNet")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--close_kernel", type=int, default=3)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def postprocess_and_extract_components(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    close_kernel_px: int = 3,
    min_area_px: int = 30,
) -> List[np.ndarray]:
    binary = (prob_map >= threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    components = []
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            component = (labels == lbl).astype(np.uint8)
            components.append(component)

    return components


def run_inference(
    model: nn.Module,
    dataset: SolarFilamentDataset,
    device: torch.device,
    tile_size: int = 256,
    overlap: float = 0.25,
    threshold: float = 0.5,
    min_area: int = 30,
    close_kernel: int = 3,
    batch_size: int = 8,
) -> Dict[str, List[str]]:
    model.eval()
    predictions = {}

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_id = sample["image_id"]
            image = sample["image"].numpy()
            valid_mask = sample["valid_mask"].numpy()

            prob_map = tiled_predict(
                model,
                image,
                valid_mask,
                tile=tile_size,
                overlap=overlap,
                device=device,
                batch_size=batch_size,
            )

            components = postprocess_and_extract_components(
                prob_map,
                threshold=threshold,
                close_kernel_px=close_kernel,
                min_area_px=min_area,
            )

            rle_strings = [binary_mask_to_rle(comp) for comp in components]
            predictions[image_id] = rle_strings

    return predictions


def main():
    args = parse_args()

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    model = MicroFilNet().to(device)

    checkpoint_path = Path(args.weights)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(str(checkpoint_path), model, device=str(device))

    test_dataset = SolarFilamentDataset(
        data_root=args.data_root,
        split="test",
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_cache=args.use_cache,
    )

    predictions = run_inference(
        model=model,
        dataset=test_dataset,
        device=device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
        min_area=args.min_area,
        close_kernel=args.close_kernel,
        batch_size=args.batch_size,
    )

    create_submission_csv(predictions, args.output)


if __name__ == "__main__":
    main()