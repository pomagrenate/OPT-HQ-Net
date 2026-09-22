from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def normalize_01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1.0, 99.0])
    denom = max(float(hi - lo), 1e-6)
    return np.clip((img - lo) / denom, 0.0, 1.0).astype(np.float32)


def fast_radial_flatten(
    img: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    n_bins: int = 128,
) -> np.ndarray:
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(float(r), 1.0)
    bin_idx = np.clip((rr * n_bins).astype(np.int32), 0, n_bins - 1)

    disk_mask = rr <= 1.0
    valid_bins = bin_idx[disk_mask]
    valid_pixels = img[disk_mask]

    if valid_pixels.size == 0:
        return normalize_01(img)

    counts = np.bincount(valid_bins, minlength=n_bins)
    sums = np.bincount(valid_bins, weights=valid_pixels, minlength=n_bins)
    profile = np.zeros(n_bins, dtype=np.float32)
    nonzero = counts > 0
    profile[nonzero] = sums[nonzero] / counts[nonzero]

    for b in range(1, n_bins):
        if counts[b] == 0:
            profile[b] = profile[b - 1]

    for b in range(n_bins - 2, -1, -1):
        if counts[b] == 0 and profile[b] == 0.0:
            profile[b] = profile[b + 1]

    profile_smooth = cv2.blur(profile.reshape(1, -1), (9, 1)).flatten()
    profile_smooth = np.clip(profile_smooth, 1e-3, None)

    background_2d = profile_smooth[bin_idx]
    flattened = img / background_2d
    flattened[~disk_mask] = 0.0
    return normalize_01(flattened)


def enhance_dark_structures_and_edges(
    img_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    u8 = (np.clip(img_flat, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(u8).astype(np.float32) / 255.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    dark_tophat = np.clip(closed.astype(np.float32) - u8.astype(np.float32), 0.0, 255.0) / 255.0

    grad_x = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    max_e = edges.max()
    if max_e > 1e-6:
        edges /= max_e

    feature_channel = np.clip(1.0 - enhanced, 0.0, 1.0).astype(np.float32)
    ridge_channel = np.clip(0.6 * dark_tophat + 0.4 * edges, 0.0, 1.0).astype(np.float32)

    return feature_channel, ridge_channel


def process_solar_observation(raw_img: np.ndarray) -> np.ndarray:
    h, w = raw_img.shape[-2:]
    cx, cy = w // 2, h // 2
    r_sun = int(0.46 * min(h, w))

    norm = normalize_01(raw_img)
    flattened = fast_radial_flatten(norm, cx, cy, r_sun)
    feat, ridge = enhance_dark_structures_and_edges(flattened)
    return np.stack([feat, ridge], axis=0).astype(np.float32)


def continuity_safe_augment(
    img: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    angle = float(rng.uniform(0.0, 360.0))
    c, h, w = img.shape
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)

    img_out = np.stack(
        [cv2.warpAffine(img[ch], rot_mat, (w, h), flags=cv2.INTER_LINEAR) for ch in range(c)]
    )
    gt_out = cv2.warpAffine(gt, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
    valid_out = np.stack(
        [cv2.warpAffine(valid[ch], rot_mat, (w, h), flags=cv2.INTER_LINEAR) for ch in range(valid.shape[0])]
    )

    if rng.random() < 0.5:
        img_out = img_out[..., ::-1].copy()
        gt_out = gt_out[..., ::-1].copy()
        valid_out = valid_out[..., ::-1].copy()

    if rng.random() < 0.5:
        img_out = img_out[..., ::-1, :].copy()
        gt_out = gt_out[..., ::-1, :].copy()
        valid_out = valid_out[..., ::-1, :].copy()

    return (
        np.ascontiguousarray(img_out, dtype=np.float32),
        np.ascontiguousarray(gt_out, dtype=np.float32),
        np.ascontiguousarray(valid_out, dtype=np.float32),
    )