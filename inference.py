from __future__ import annotations

import cv2
import numpy as np
import torch


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    image_2048: np.ndarray,
    valid_mask: np.ndarray,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    model.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(dev)

    if image_2048.ndim == 2:
        img_tensor = torch.from_numpy(image_2048[np.newaxis, np.newaxis, ...]).float().to(dev)
    elif image_2048.ndim == 3 and image_2048.shape[0] == 1:
        img_tensor = torch.from_numpy(image_2048[np.newaxis, ...]).float().to(dev)
    else:
        raise ValueError(f"Expected shape (2048, 2048) or (1, 2048, 2048), got {image_2048.shape}")

    out = model(img_tensor)
    mask_logits = out[0] if isinstance(out, tuple) else out
    probs = torch.sigmoid(mask_logits.float())[0, 0].detach().cpu().numpy()

    if valid_mask.ndim == 3:
        valid_mask = valid_mask[0]
    probs = probs * valid_mask.astype(np.float32)

    return probs.astype(np.float32)


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


def extract_components_from_binary(binary: np.ndarray, min_area_px: int = 15) -> list[np.ndarray]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            component = (labels == lbl).astype(np.uint8)
            components.append(component)
    return components