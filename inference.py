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
    global_image: np.ndarray,
    disk_center: tuple[int, int, int],
    tile: int = 512,
    overlap: float = 0.25,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    context_margin: float = 0.5,
) -> np.ndarray:
    """
    context_margin must match the value SolarFilamentDataset was trained
    with (default 0.5 in both places) — it controls how far past each
    tile's own footprint the global-context crop extends. A mismatch here
    doesn't break shapes (spatial_align_crop is fully general), but it does
    mean the model sees a differently-scaled context window than it trained
    on, which can quietly hurt near-tile-edge predictions.
    """
    model.eval()
    dev = torch.device(device)
    c, h, w = image.shape
    cx, cy, r_sun = disk_center
    stride = max(1, int(tile * (1.0 - overlap)))

    pad_h = (-(h - tile)) % stride if h > tile else tile - h
    pad_w = (-(w - tile)) % stride if w > tile else tile - w

    img_p = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    hp, wp = img_p.shape[1:]

    img_tensor = torch.from_numpy(img_p).float()
    global_tensor = torch.from_numpy(global_image).float().unsqueeze(0).to(dev)
    weight_kernel = torch.from_numpy(_gaussian_weight(tile)).to(dev)

    accum = torch.zeros((hp, wp), dtype=torch.float32, device=dev)
    weight_accum = torch.zeros((hp, wp), dtype=torch.float32, device=dev)

    ys = list(range(0, hp - tile + 1, stride))
    xs = list(range(0, wp - tile + 1, stride))
    coords_list = [(y, x) for y in ys for x in xs]

    margin_px = context_margin * tile

    for i in range(0, len(coords_list), batch_size):
        batch_coords = coords_list[i : i + batch_size]
        batch_tensors = [img_tensor[:, y : y + tile, x : x + tile] for y, x in batch_coords]
        batch_local = torch.stack(batch_tensors, dim=0).to(dev, non_blocking=True)

        batch_meta = []
        batch_bbox = []
        for y, x in batch_coords:
            center_x = x + tile / 2.0
            center_y = y + tile / 2.0
            x_norm = center_x / float(w)
            y_norm = center_y / float(h)
            r_norm = np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2) / max(float(r_sun), 1.0)
            batch_meta.append([x_norm, y_norm, r_norm])

            x0c = (x - margin_px) / float(w)
            y0c = (y - margin_px) / float(h)
            x1c = (x + tile + margin_px) / float(w)
            y1c = (y + tile + margin_px) / float(h)
            batch_bbox.append([x0c, y0c, x1c, y1c])

        batch_meta_t = torch.tensor(batch_meta, dtype=torch.float32, device=dev)
        batch_bbox_t = torch.tensor(batch_bbox, dtype=torch.float32, device=dev)

        batch_global = global_tensor.expand(batch_local.size(0), -1, -1, -1)

        logits = model(batch_local, batch_global, batch_meta_t, batch_bbox_t)
        probs = torch.sigmoid(logits.float())[:, 0]

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    valid_labels = np.where((areas >= min_area_px) & (np.arange(n_labels) > 0))[0]
    return np.isin(labels, valid_labels).astype(np.uint8)
