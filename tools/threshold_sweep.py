"""
Threshold Sweep & Post-Processing Evaluator for Filament-HQ.

Evaluates a trained checkpoint (e.g., epoch 18 or best_model.pt) across multiple
semantic thresholds [0.20 ... 0.80] to isolate model quality from instance grouping sensitivity.

Usage Example:
  python tools/threshold_sweep.py \
    --weights checkpoints_hq/best_model.pt \
    --data_root /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train \
    --cache /kaggle/working/magfilo_hq_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from filament_hq.data.dataset import FilamentTileDataset, filament_hq_collate_fn
from filament_hq.metrics.panoptic_quality import PanopticQualityMetric
from filament_hq.models.model import FilamentHQModel
from filament_hq.postprocessing.grouping import FilamentPostProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Filament-HQ Threshold Sweep & Post-Processing Evaluator")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint weights (.pt)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--cache", type=str, default=None, help="Path to precomputed cache directory")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone model architecture")
    parser.add_argument("--imgsz", type=int, default=1024, help="Tile input size")
    parser.add_argument("--batch_size", type=int, default=2, help="Validation batch size")
    return parser.parse_args()


@torch.no_grad()
def run_sweep():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    print(f"\n[Sweep] Loading model with backbone '{args.backbone}' from '{args.weights}'...")
    model = FilamentHQModel(backbone=args.backbone)
    state_dict = torch.load(args.weights, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device).eval()

    # Load Validation Dataset
    val_ds = FilamentTileDataset(
        data_root=args.data_root,
        tile_size=args.imgsz,
        fg_prob=0.0,
        augment=False,
        cache_dir=args.cache,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=filament_hq_collate_fn,
    )
    print(f"[Sweep] Evaluating on {len(val_ds)} validation tiles...")

    threshold_candidates = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    results = []

    for thresh in threshold_candidates:
        postprocessor = FilamentPostProcessor(
            seed_thresh=min(thresh + 0.15, 0.90),
            mask_thresh=thresh,
            min_area=50,
        )
        metric = PanopticQualityMetric(iou_threshold=0.5)

        total_gt = 0
        total_pred = 0

        for batch in tqdm(val_loader, desc=f"Sweep Thresh={thresh:.2f}", leave=False):
            images = batch["image"].to(device)
            outputs = model(images)

            sem_probs = torch.sigmoid(outputs["semantic"]).cpu().numpy()
            bnd_probs = torch.sigmoid(outputs["boundary"]).cpu().numpy() if "boundary" in outputs else None
            skl_probs = torch.sigmoid(outputs["skeleton"]).cpu().numpy() if "skeleton" in outputs else None

            pred_np = []
            gt_np = []

            for b in range(images.shape[0]):
                if "instance_masks" in batch and isinstance(batch["instance_masks"], list):
                    gt_arr = batch["instance_masks"][b]
                    if isinstance(gt_arr, torch.Tensor):
                        gt_arr = gt_arr.cpu().numpy()
                else:
                    g_mask = (batch["semantic"][b, 0].cpu().numpy() > 0.5).astype(np.uint8)
                    num_g, g_labels = cv2.connectedComponents(g_mask)
                    gt_arr = np.stack([(g_labels == j).astype(np.uint8) for j in range(1, num_g)], axis=0) if num_g > 1 else np.zeros((0, 1024, 1024), dtype=np.uint8)

                b_bnd = bnd_probs[b, 0] if bnd_probs is not None else None
                b_skl = skl_probs[b, 0] if skl_probs is not None else None

                pred_arr = postprocessor.process(
                    sem_prob=sem_probs[b, 0],
                    bnd_prob=b_bnd,
                    skl_prob=b_skl,
                )

                pred_np.append(pred_arr)
                gt_np.append(gt_arr)

                total_pred += len(pred_arr)
                total_gt += len(gt_arr)

            metric.update(pred_np, gt_np)

        m = metric.compute()
        tp, fp, fn = m.get("TP", 0), m.get("FP", 0), m.get("FN", 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        ratio = total_pred / max(total_gt, 1)

        results.append({
            "thresh": thresh,
            "pred": total_pred,
            "gt": total_gt,
            "ratio": ratio,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "dice": m.get("mean_dice", 0.0),
            "pq": m.get("PQ", 0.0),
            "sq": m.get("SQ", 0.0),
            "rq": m.get("RQ", 0.0),
            "prec": precision,
            "rec": recall,
        })

    # Print Summary Results Table
    print("\n===========================================================================================================")
    print("                              📊 FILAMENT-HQ THRESHOLD SWEEP RESULTS TABLE                                 ")
    print("===========================================================================================================")
    print(f" {'Thresh':<7} | {'Pred':<6} | {'GT':<6} | {'Ratio':<6} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Dice':<7} | {'PQ':<7} | {'SQ':<7} | {'RQ':<7} | {'Prec':<6} | {'Rec':<6}")
    print("-" * 107)

    best_res = max(results, key=lambda x: x["pq"])

    for r in results:
        marker = " 🏆" if r["thresh"] == best_res["thresh"] else ""
        print(
            f" {r['thresh']:<7.2f} | {r['pred']:<6d} | {r['gt']:<6d} | {r['ratio']:<6.2f} | "
            f"{r['tp']:<5d} | {r['fp']:<5d} | {r['fn']:<5d} | {r['dice']:<7.4f} | {r['pq']:<7.4f} | "
            f"{r['sq']:<7.4f} | {r['rq']:<7.4f} | {r['prec']:<6.4f} | {r['rec']:<6.4f}{marker}"
        )
    print("===========================================================================================================\n")
    print(f"🎯 OPTIMAL POST-PROCESSING THRESHOLD: {best_res['thresh']:.2f} (Peak PQ = {best_res['pq']:.4f}, Dice = {best_res['dice']:.4f})\n")


if __name__ == "__main__":
    run_sweep()
