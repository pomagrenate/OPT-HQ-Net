from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

try:
    from skimage.filters import frangi
    from skimage.morphology import black_tophat, disk as skimage_disk
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def load_fits_as_array(path: str) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data


def normalize_01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    denom = max(float(hi - lo), 1e-6)
    return np.clip((img - lo) / denom, 0.0, 1.0).astype(np.float32)


def detect_solar_disk(img_u8: np.ndarray) -> tuple[int, int, int]:
    h, w = img_u8.shape
    blurred = cv2.GaussianBlur(img_u8, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=float(w),
        param1=50,
        param2=30,
        minRadius=int(0.35 * min(h, w)),
        maxRadius=int(0.5 * min(h, w)),
    )
    if circles is not None:
        cx, cy, r = circles[0, 0]
        return int(round(cx)), int(round(cy)), int(round(r))
    return w // 2, h // 2, int(0.45 * min(h, w))


def make_disk_mask(
    shape: tuple[int, int],
    cx: int,
    cy: int,
    r: int,
    shrink_px: int = 2,
) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (dist2 <= (max(r - shrink_px, 0)) ** 2).astype(np.float32)


def radial_flatten(
    img: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    n_bins: int = 200,
) -> np.ndarray:
    h, w = img.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(float(r), 1.0)
    bin_idx = np.clip((rr * n_bins).astype(np.int32), 0, n_bins - 1)

    valid = mask > 0.5
    valid_bins = bin_idx[valid]
    valid_pixels = img[valid]

    if len(valid_pixels) == 0:
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

    k = max(3, (n_bins // 40) | 1)
    profile_smooth = cv2.blur(profile.reshape(1, -1), (k, 1)).flatten()
    profile_smooth = np.clip(profile_smooth, 1e-3, None)

    background_2d = profile_smooth[bin_idx]
    flattened = img / background_2d
    median_val = float(np.median(valid_pixels)) if valid_pixels.size > 0 else 1.0
    flattened *= median_val
    flattened[~valid] = 0.0
    return normalize_01(flattened)


def denoise_and_enhance(
    img: np.ndarray,
    mask: np.ndarray,
    clahe_clip: float = 2.5,
    clahe_tile: int = 8,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0.7)
    u8 = (np.clip(blurred, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(u8).astype(np.float32) / 255.0
    enhanced[mask < 0.5] = 0.0
    return enhanced


def ridge_prior(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inv = 1.0 - img
    if _HAS_SKIMAGE:
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        th = black_tophat(u8, footprint=skimage_disk(9)).astype(np.float32) / 255.0
        vess = frangi(inv, sigmas=range(1, 4), black_ridges=False)
        vess = vess / (vess.max() + 1e-6)
        prior = np.clip(0.5 * th + 0.5 * vess, 0.0, 1.0)
    else:
        u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
        th = np.clip(closed.astype(np.float32) - u8, 0.0, 255.0) / 255.0
        prior = th

    prior[mask < 0.5] = 0.0
    return prior.astype(np.float32)


@dataclass
class PreprocessedObservation:
    image: np.ndarray
    valid_mask: np.ndarray
    disk: tuple[int, int, int]


def preprocess_observation(raw: np.ndarray) -> PreprocessedObservation:
    img = normalize_01(raw)
    u8 = (img * 255.0).astype(np.uint8)
    cx, cy, r = detect_solar_disk(u8)
    mask = make_disk_mask(img.shape, cx, cy, r)

    flattened = radial_flatten(img, mask, cx, cy, r)
    enhanced = denoise_and_enhance(flattened, mask)
    prior = ridge_prior(enhanced, mask)

    stack = np.stack([enhanced, prior], axis=0).astype(np.float32)
    return PreprocessedObservation(
        image=stack,
        valid_mask=mask[None, ...].astype(np.float32),
        disk=(cx, cy, r),
    )


def extract_tiles(
    image: np.ndarray,
    mask_valid: np.ndarray,
    gt_mask: np.ndarray | None,
    tile: int = 256,
    overlap: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, bool, tuple[int, int]]]:
    c, h, w = image.shape
    stride = max(1, int(tile * (1.0 - overlap)))
    tiles = []

    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            valid_t = mask_valid[:, y : y + tile, x : x + tile]
            if valid_t.mean() < 0.05:
                continue

            img_t = image[:, y : y + tile, x : x + tile]
            gt_t = None
            has_fil = False

            if gt_mask is not None:
                gt_t = gt_mask[y : y + tile, x : x + tile]
                has_fil = bool(gt_t.sum() > 0)

            tiles.append((img_t, valid_t, gt_t, has_fil, (y, x)))

    return tiles


def class_balanced_sample(
    tiles: list,
    positive_ratio: float = 0.7,
    n: int | None = None,
    rng: np.random.Generator | None = None,
) -> list:
    rng = rng or np.random.default_rng()
    pos = [t for t in tiles if t[3]]
    neg = [t for t in tiles if not t[3]]
    target_n = n or len(tiles)
    n_pos = int(target_n * positive_ratio)
    n_neg = target_n - n_pos

    if len(pos) == 0:
        pos_sample = []
        n_neg = target_n
    else:
        replace_pos = len(pos) < n_pos
        pos_indices = rng.choice(len(pos), size=n_pos, replace=replace_pos)
        pos_sample = [pos[i] for i in pos_indices]

    if len(neg) == 0:
        neg_sample = []
    else:
        replace_neg = len(neg) < n_neg
        neg_indices = rng.choice(len(neg), size=n_neg, replace=replace_neg)
        neg_sample = [neg[i] for i in neg_indices]

    combined = pos_sample + neg_sample
    rng.shuffle(combined)
    return combined


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

    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-0.05, 0.05))
    img_out[0] = np.clip(img_out[0] * gain + bias, 0.0, 1.0)

    return (
        np.ascontiguousarray(img_out, dtype=np.float32),
        np.ascontiguousarray(gt_out, dtype=np.float32),
        np.ascontiguousarray(valid_out, dtype=np.float32),
    )