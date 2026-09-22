from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def binary_mask_to_rle(mask: np.ndarray) -> str:
    mask_flat = np.asfortranarray(mask.astype(bool)).flatten()
    if not np.any(mask_flat):
        return ""

    pixels = np.empty(mask_flat.size + 2, dtype=np.int8)
    pixels[0] = 0
    pixels[1:-1] = mask_flat
    pixels[-1] = 0

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_to_binary_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    if not rle_str or pd.isna(rle_str):
        return np.zeros((height, width), dtype=np.uint8)

    runs = np.fromstring(rle_str, dtype=int, sep=" ")
    starts = runs[0::2] - 1
    lengths = runs[1::2]
    ends = starts + lengths

    mask_flat = np.zeros(height * width, dtype=np.uint8)
    for s, e in zip(starts, ends):
        mask_flat[s:e] = 1

    return mask_flat.reshape((width, height), order="F").T


def compute_panoptic_quality(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "TP": 0, "FP": 0, "FN": 0}
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "TP": 0, "FP": len(pred_masks), "FN": len(gt_masks)}

    def iou(m1: np.ndarray, m2: np.ndarray) -> float:
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(inter / (union + 1e-8))

    matched_gt = set()
    iou_sum = 0.0
    tp = 0

    for p in pred_masks:
        best_iou = 0.0
        best_idx = -1
        for j, g in enumerate(gt_masks):
            if j in matched_gt:
                continue
            curr_iou = iou(p, g)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_idx = j

        if best_iou >= iou_threshold and best_idx != -1:
            matched_gt.add(best_idx)
            iou_sum += best_iou
            tp += 1

    fp = len(pred_masks) - tp
    fn = len(gt_masks) - len(matched_gt)

    sq = iou_sum / (tp + 1e-8) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
    return {
        "PQ": sq * rq,
        "SQ": sq,
        "RQ": rq,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.decay = decay
        self.device = device
        self.shadow_model = copy.deepcopy(model).eval()
        if device is not None:
            self.shadow_model.to(device)

        for p in self.shadow_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for s_param, m_param in zip(self.shadow_model.parameters(), model.parameters()):
            s_param.data.mul_(d).add_(m_param.data.to(s_param.device), alpha=(1.0 - d))

        for s_buf, m_buf in zip(self.shadow_model.buffers(), model.buffers()):
            s_buf.copy_(m_buf.to(s_buf.device))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow_state_dict": self.shadow_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.decay = state_dict["decay"]
        self.shadow_model.load_state_dict(state_dict["shadow_state_dict"])


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    ema_model: Optional[ModelEMA] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):
    ckpt = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if ema_model is not None:
        ckpt["ema_state_dict"] = ema_model.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(ckpt, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[ModelEMA] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ema_model is not None and "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return {"epoch": ckpt.get("epoch", 0), "loss": ckpt.get("loss", float("inf"))}


def create_submission_csv(predictions: Dict[str, List[str]], output_path: str):
    rows = []
    for image_id, rle_list in predictions.items():
        if not rle_list:
            rows.append({"filament_id": f"{image_id}_none", "segmentation_rle": ""})
            continue
        for i, rle_str in enumerate(rle_list):
            rows.append({
                "filament_id": f"{image_id}_{i + 1}",
                "segmentation_rle": rle_str,
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)