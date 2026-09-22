from __future__ import annotations

import cv2
import numpy as np
import torch


def _gaussian_weight(tile: int, sigma_frac: float = 0.5) -> np.ndarray:
    ax = np.linspace(-1.0, 1.0, tile, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    d2 = xx ** 2 + yy ** 2
    w = np.exp(-d2 / (2.0 * (sigma_frac ** 2)))
    return w.astype(np.float32)


@torch.no_grad()
def tiled_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    valid_mask: np.ndarray,
    tile: int = 256,
    overlap: float = 0.25,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    c, h, w = image.shape
    stride = max(1, int(tile * (1.0 - overlap)))

    pad_h = (-(h - tile)) % stride if h > tile else tile - h
    pad_w = (-(w - tile)) % stride if w > tile else tile - w

    img_p = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    hp, wp = img_p.shape[1:]

    img_tensor = torch.from_numpy(img_p).float()
    weight_kernel = torch.from_numpy(_gaussian_weight(tile)).to(dev)

    accum = torch.zeros((hp, wp), dtype=torch.float32, device=dev)
    weight_accum = torch.zeros((hp, wp), dtype=torch.float32, device=dev)

    ys = list(range(0, hp - tile + 1, stride))
    xs = list(range(0, wp - tile + 1, stride))
    coords = [(y, x) for y in ys for x in xs]

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        batch_tensors = [
            img_tensor[:, y : y + tile, x : x + tile] for y, x in batch_coords
        ]
        batch = torch.stack(batch_tensors, dim=0).to(dev, non_blocking=True)

        logits = model(batch)
        probs = torch.sigmoid(logits)[:, 0]

        for idx, (y, x) in enumerate(batch_coords):
            accum[y : y + tile, x : x + tile] += probs[idx] * weight_kernel
            weight_accum[y : y + tile, x : x + tile] += weight_kernel

    weight_accum = torch.clamp(weight_accum, min=1e-6)
    full_prob = accum / weight_accum
    full_prob = full_prob[:h, :w]

    valid_mask_t = torch.from_numpy(valid_mask[0]).to(dev)
    full_prob = full_prob * valid_mask_t

    return full_prob.detach().cpu().numpy().astype(np.float32)


def postprocess_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    close_kernel_px: int = 3,
    min_area_px: int = 15,
) -> np.ndarray:
    binary = (prob_map >= threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    areas = stats[:, cv2.CC_STAT_AREA]
    valid_labels = np.where((areas >= min_area_px) & (np.arange(n_labels) > 0))[0]

    mask = np.isin(labels, valid_labels).astype(np.uint8)
    return mask


def predict_and_clean(
    model: torch.nn.Module,
    image: np.ndarray,
    valid_mask: np.ndarray,
    tile: int = 256,
    overlap: float = 0.25,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    prob_map = tiled_predict(model, image, valid_mask, tile, overlap, device)
    mask = postprocess_mask(prob_map, threshold=threshold)
    return {"probability": prob_map, "mask": mask}