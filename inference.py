"""
Tiled inference + postprocessing for MicroFilNet.

Why tile at all: full-disk GONG frames are far larger than a training crop,
and re-detection every tile boundary is exactly where fragmentation tends
to be introduced. So:

  1. Overlapping tiles, each predicted independently.
  2. Reassembled with a Gaussian-weighted blend (higher weight near tile
     center, tapering to the edges) so no hard seams appear in the
     probability map -- this alone removes most boundary-induced breaks.
  3. Zero-parameter morphological cleanup as a final, cheap continuity
     pass: small-gap closing (bridges 1-2px breaks in barbs) and
     area-based pruning of isolated speckle, both tuned to typical
     filament/barb widths so real thin structures are not eroded away.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError as e:
    raise ImportError("inference.py needs opencv-python") from e


def _gaussian_weight(tile: int, sigma_frac: float = 0.5) -> np.ndarray:
    ax = np.linspace(-1, 1, tile)
    xx, yy = np.meshgrid(ax, ax)
    d = np.sqrt(xx ** 2 + yy ** 2)
    sigma = sigma_frac
    w = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return w.astype(np.float32)


@torch.no_grad()
def tiled_predict(model, image: np.ndarray, valid_mask: np.ndarray,
                   tile: int = 256, overlap: float = 0.25,
                   device: str = "cpu", batch_size: int = 8) -> np.ndarray:
    """
    image: (2, H, W) preprocessed float32 array (see preprocessing.py)
    valid_mask: (1, H, W)
    Returns: (H, W) float32 probability map in [0, 1], seam-free.
    """
    model.eval()
    C, H, W = image.shape
    stride = max(1, int(tile * (1 - overlap)))
    weight_kernel = _gaussian_weight(tile)

    # pad so tiles cover the full frame exactly
    pad_h = (-(H - tile)) % stride if H > tile else tile - H
    pad_w = (-(W - tile)) % stride if W > tile else tile - W
    img_p = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = img_p.shape[1:]

    accum = np.zeros((Hp, Wp), dtype=np.float32)
    weight_accum = np.zeros((Hp, Wp), dtype=np.float32)

    ys = list(range(0, Hp - tile + 1, stride))
    xs = list(range(0, Wp - tile + 1, stride))
    coords = [(y, x) for y in ys for x in xs]

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        batch = np.stack([img_p[:, y:y + tile, x:x + tile] for y, x in batch_coords])
        batch_t = torch.from_numpy(batch).to(device)
        logits = model(batch_t)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (b, tile, tile)
        for (y, x), p in zip(batch_coords, probs):
            accum[y:y + tile, x:x + tile] += p * weight_kernel
            weight_accum[y:y + tile, x:x + tile] += weight_kernel

    weight_accum = np.clip(weight_accum, 1e-6, None)
    full_prob = accum / weight_accum
    full_prob = full_prob[:H, :W]
    full_prob = full_prob * valid_mask[0]
    return full_prob.astype(np.float32)


def postprocess_mask(prob_map: np.ndarray, threshold: float = 0.5,
                      close_kernel_px: int = 3, min_area_px: int = 15) -> np.ndarray:
    """
    Zero-parameter, topology-preserving cleanup:
      - binary closing with a small kernel sized to typical barb width:
        bridges tiny gaps without fattening/merging distinct filaments
      - connected-component area filtering: drops isolated speckle noise,
        but only components below `min_area_px`, so real thin barbs survive
    """
    binary = (prob_map >= threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_px, close_kernel_px))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == lbl] = 1
    return cleaned


def predict_and_clean(model, image: np.ndarray, valid_mask: np.ndarray,
                       tile: int = 256, overlap: float = 0.25, device: str = "cpu",
                       threshold: float = 0.5) -> dict:
    prob_map = tiled_predict(model, image, valid_mask, tile, overlap, device)
    mask = postprocess_mask(prob_map, threshold=threshold)
    return {"probability": prob_map, "mask": mask}
