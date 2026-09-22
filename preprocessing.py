"""
MicroFilNet preprocessing pipeline
===================================
Raw GONG H-alpha FITS  ->  model-ready tensor stack.

Pipeline stages (see README.md Section 1 for the full derivation):
  A. Geometric standardization  : load FITS, normalize, detect disk, build valid-pixel mask
  B. Radiometric correction     : radial limb-darkening flattening, denoise, CLAHE
  C. Zero-parameter physics priors : black top-hat + multiscale Frangi "vesselness" ridge map
  D. Tiling                     : overlapping patch extraction with class-balanced sampling

Output of `preprocess_observation()` is a dict with:
  - "image"       : (2, H, W) float32   channel 0 = corrected H-alpha, channel 1 = ridge prior
  - "valid_mask"  : (1, H, W) float32   1.0 on-disk, 0.0 off-disk (multiply into every loss)
  - "disk"        : (cx, cy, r) in pixels, for bookkeeping / reprojecting predictions

No learned parameters are used anywhere in this file -- every step is classical
image processing, so it costs nothing out of the model's parameter budget while
supplying strong orientation/continuity priors "for free".
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "This module needs opencv-python (`pip install opencv-python --break-system-packages`)"
    ) from e

try:
    from skimage.filters import frangi
    from skimage.morphology import black_tophat, disk as skimage_disk
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# --------------------------------------------------------------------------
# A. Geometric standardization
# --------------------------------------------------------------------------

def load_fits_as_array(path: str) -> np.ndarray:
    """Load a GONG H-alpha FITS file as a float32 array. Requires astropy."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data


def normalize_01(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return img.astype(np.float32)


def detect_solar_disk(img_u8: np.ndarray) -> tuple[int, int, int]:
    """
    Detect (center_x, center_y, radius) of the solar disk via Hough circles.
    Falls back to an image-centered disk if detection fails (robust default
    for full-disk GONG frames, which are already disk-centered by design).
    """
    h, w = img_u8.shape
    blurred = cv2.GaussianBlur(img_u8, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w,
        param1=50, param2=30,
        minRadius=int(0.35 * min(h, w)), maxRadius=int(0.5 * min(h, w)),
    )
    if circles is not None:
        cx, cy, r = circles[0, 0]
        return int(round(cx)), int(round(cy)), int(round(r))
    # fallback: assume centered disk filling ~90% of the frame
    cx, cy = w // 2, h // 2
    r = int(0.45 * min(h, w))
    return cx, cy, r


def make_disk_mask(shape: tuple[int, int], cx: int, cy: int, r: int, shrink_px: int = 2) -> np.ndarray:
    """Binary mask, 1 inside the disk (shrunk by a couple of px to kill limb ringing)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 <= (r - shrink_px) ** 2).astype(np.float32)
    return mask


# --------------------------------------------------------------------------
# B. Radiometric correction
# --------------------------------------------------------------------------

def radial_flatten(img: np.ndarray, mask: np.ndarray, cx: int, cy: int, r: int,
                    n_bins: int = 200) -> np.ndarray:
    """
    Remove center-to-limb variation (limb darkening) by estimating and dividing
    out a smooth radial brightness profile, computed only from on-disk pixels.
    """
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1)   # normalized radius in [0,1] on-disk

    bin_edges = np.linspace(0, 1.0, n_bins + 1)
    bin_idx = np.clip((rr * n_bins).astype(int), 0, n_bins - 1)

    profile = np.zeros(n_bins, dtype=np.float32)
    valid = mask > 0.5
    for b in range(n_bins):
        sel = valid & (bin_idx == b)
        if sel.sum() > 10:
            profile[b] = np.median(img[sel])
        elif b > 0:
            profile[b] = profile[b - 1]

    # smooth the 1-D radial profile before dividing it out
    k = max(3, n_bins // 40 | 1)  # odd kernel
    profile_smooth = cv2.blur(profile.reshape(1, -1), (k, 1)).flatten()
    profile_smooth = np.clip(profile_smooth, 1e-3, None)

    background_2d = profile_smooth[bin_idx]
    flattened = img / background_2d
    flattened = flattened * float(np.median(img[valid]))  # rescale back to original brightness range
    flattened[~valid] = 0.0
    return normalize_01(flattened)


def denoise_and_enhance(img: np.ndarray, mask: np.ndarray,
                         clahe_clip: float = 2.5, clahe_tile: int = 8) -> np.ndarray:
    """Gaussian denoise (kept small so 1-2px barbs survive) + CLAHE local contrast."""
    blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0.7)
    u8 = (np.clip(blurred, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    enhanced = clahe.apply(u8).astype(np.float32) / 255.0
    enhanced[mask < 0.5] = 0.0
    return enhanced


# --------------------------------------------------------------------------
# C. Zero-parameter ridge / continuity priors
# --------------------------------------------------------------------------

def ridge_prior(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Classical, non-learned prior that already "knows" filaments are dark,
    thin, elongated structures. Combines:
      - black top-hat (highlights small dark structures on a bright background)
      - multiscale Frangi vesselness on the *inverted* image (filaments are dark
        ridges, Frangi expects bright ridges, so invert first)
    This channel gives the network orientation/continuity information for
    zero learned parameters -- it is concatenated to the raw image as input
    channel 2 and is also fed (unlearned) into the bottleneck FiLM modulation.
    """
    inv = 1.0 - img
    if _HAS_SKIMAGE:
        th = black_tophat((img * 255).astype(np.uint8), footprint=skimage_disk(9)).astype(np.float32) / 255.0
        vess = frangi(inv, sigmas=range(1, 4), black_ridges=False)
        vess = vess / (vess.max() + 1e-6)
        prior = np.clip(0.5 * th + 0.5 * vess, 0, 1)
    else:
        # fallback with pure OpenCV if scikit-image isn't installed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        closed = cv2.morphologyEx((img * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        th = np.clip(closed.astype(np.float32) - (img * 255), 0, 255) / 255.0
        prior = th
    prior[mask < 0.5] = 0.0
    return prior.astype(np.float32)


# --------------------------------------------------------------------------
# Full observation-level pipeline
# --------------------------------------------------------------------------

@dataclass
class PreprocessedObservation:
    image: np.ndarray        # (2, H, W) float32
    valid_mask: np.ndarray   # (1, H, W) float32
    disk: tuple[int, int, int]


def preprocess_observation(raw: np.ndarray) -> PreprocessedObservation:
    img = normalize_01(raw)
    u8 = (img * 255).astype(np.uint8)
    cx, cy, r = detect_solar_disk(u8)
    mask = make_disk_mask(img.shape, cx, cy, r)

    flattened = radial_flatten(img, mask, cx, cy, r)
    enhanced = denoise_and_enhance(flattened, mask)
    prior = ridge_prior(enhanced, mask)

    stack = np.stack([enhanced, prior], axis=0)  # (2, H, W)
    return PreprocessedObservation(
        image=stack.astype(np.float32),
        valid_mask=mask[None, ...].astype(np.float32),
        disk=(cx, cy, r),
    )


# --------------------------------------------------------------------------
# D. Tiling / class-balanced patch sampling / continuity-safe augmentation
# --------------------------------------------------------------------------

def extract_tiles(image: np.ndarray, mask_valid: np.ndarray, gt_mask: np.ndarray | None,
                   tile: int = 256, overlap: float = 0.25):
    """
    Yield overlapping tiles (for tiled inference) or, if gt_mask is given,
    (tile_image, tile_valid, tile_gt, has_filament) for training sampling.
    """
    C, H, W = image.shape
    stride = int(tile * (1 - overlap))
    tiles = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            img_t = image[:, y:y + tile, x:x + tile]
            valid_t = mask_valid[:, y:y + tile, x:x + tile]
            if valid_t.mean() < 0.05:      # skip pure off-disk tiles
                continue
            gt_t = None
            has_fil = False
            if gt_mask is not None:
                gt_t = gt_mask[y:y + tile, x:x + tile]
                has_fil = gt_t.sum() > 0
            tiles.append((img_t, valid_t, gt_t, has_fil, (y, x)))
    return tiles


def class_balanced_sample(tiles: list, positive_ratio: float = 0.7, n: int | None = None,
                           rng: np.random.Generator | None = None) -> list:
    """
    Oversample filament-containing tiles. `positive_ratio` fraction of the
    sampled set will contain at least one filament pixel.
    """
    rng = rng or np.random.default_rng()
    pos = [t for t in tiles if t[3]]
    neg = [t for t in tiles if not t[3]]
    n = n or len(tiles)
    n_pos = int(n * positive_ratio)
    n_neg = n - n_pos
    if len(pos) == 0:
        pos_sample = []
        n_neg = n
    else:
        pos_sample = list(rng.choice(len(pos), size=min(n_pos, len(pos) * 5), replace=len(pos) < n_pos))
        pos_sample = [pos[i] for i in pos_sample]
    neg_sample = list(rng.choice(len(neg), size=min(n_neg, max(len(neg), 1)), replace=len(neg) < n_neg)) if neg else []
    neg_sample = [neg[i] for i in neg_sample]
    combined = pos_sample + neg_sample
    rng.shuffle(combined)
    return combined


def continuity_safe_augment(img: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                             rng: np.random.Generator | None = None):
    """
    Augmentations chosen specifically to NOT sever thin barbs/spines:
      - free rotation (filaments have no canonical orientation)
      - H/V flip
      - mild elastic deformation
      - brightness/contrast jitter
    Deliberately EXCLUDES cutout/random-erasing, which can cut a filament
    into two disconnected pieces and would directly fight the continuity
    objective (soft-clDice) used at training time.
    """
    rng = rng or np.random.default_rng()
    angle = rng.uniform(0, 360)
    C, H, W = img.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)

    def rot(x, is_mask):
        flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        if x.ndim == 3:
            out = np.stack([cv2.warpAffine(x[c], M, (W, H), flags=flags) for c in range(x.shape[0])])
        else:
            out = cv2.warpAffine(x, M, (W, H), flags=flags)
        return out

    img_r, gt_r, valid_r = rot(img, False), rot(gt, True), rot(valid, False)

    if rng.random() < 0.5:
        img_r, gt_r, valid_r = img_r[..., ::-1].copy(), gt_r[..., ::-1].copy(), valid_r[..., ::-1].copy()
    if rng.random() < 0.5:
        img_r, gt_r, valid_r = img_r[..., ::-1, :].copy(), gt_r[..., ::-1, :].copy(), valid_r[..., ::-1, :].copy()

    # brightness/contrast jitter on the raw-intensity channel only (channel 0)
    gain = rng.uniform(0.85, 1.15)
    bias = rng.uniform(-0.05, 0.05)
    img_r[0] = np.clip(img_r[0] * gain + bias, 0, 1)

    return img_r.astype(np.float32), gt_r.astype(np.float32), valid_r.astype(np.float32)
