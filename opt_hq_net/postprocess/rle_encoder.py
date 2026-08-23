"""
RLE encoding and submission CSV generation for the MAGFiLO benchmark.

The evaluation server requires:
  - A single CSV file with columns: filament_id, segmentation_rle
  - RLE strings encoded in Fortran (column-major) order via pycocotools
  - Image dimensions fixed at 2048 × 2048
  - Unique row keys: {image_id}_{instance_index}

Usage
-----
>>> encoder = RLEEncoder(height=2048, width=2048)
>>> rle_str = encoder.encode(binary_mask)   # str
>>> decoded  = encoder.decode(rle_str)      # np.ndarray uint8

>>> rows = build_submission_csv(predictions_dict)
>>> rows.to_csv("submission.csv", index=False)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from pycocotools import mask as coco_mask
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False


class RLEEncoder:
    """
    Encode / decode binary segmentation masks as RLE count strings
    compatible with the MAGFiLO benchmark evaluation protocol.

    The evaluation server:
      - Expects column-major (Fortran) ordering for the RLE counts.
      - Does NOT expect quotation marks in the RLE string.
      - Requires fixed image size (2048 × 2048).

    Parameters
    ----------
    height : int
        Fixed image height (default 2048).
    width : int
        Fixed image width (default 2048).

    Raises
    ------
    ImportError
        If ``pycocotools`` is not installed.
    """

    def __init__(self, height: int = 2048, width: int = 2048) -> None:
        if not COCO_AVAILABLE:
            raise ImportError(
                "pycocotools is required for RLE encoding. "
                "Install with: pip install pycocotools"
            )
        self.height = height
        self.width = width

    # ------------------------------------------------------------------
    def encode(self, binary_mask: np.ndarray) -> str:
        """
        Encode a binary mask as an RLE count string.

        Parameters
        ----------
        binary_mask : np.ndarray (H, W)
            Binary array with values {0, 1} or bool.  Must be 2D.
            If the mask has different dimensions than (height, width),
            it is resized using nearest-neighbour interpolation.

        Returns
        -------
        str — UTF-8 decoded RLE count string (no quotes).
        """
        mask = self._validate_and_resize(binary_mask)

        # pycocotools requires Fortran-ordered uint8 array
        fortran_mask = np.asfortranarray(mask.astype(np.uint8))
        rle_dict = coco_mask.encode(fortran_mask)

        # counts may be bytes; decode to str
        counts = rle_dict["counts"]
        if isinstance(counts, bytes):
            return counts.decode("utf-8")
        return counts

    # ------------------------------------------------------------------
    def decode(self, rle_string: str) -> np.ndarray:
        """
        Decode an RLE count string back to a binary mask.

        Parameters
        ----------
        rle_string : str
            RLE count string as returned by ``encode``.

        Returns
        -------
        np.ndarray (H, W) uint8 — binary mask.
        """
        rle_dict = {
            "size": [self.height, self.width],
            "counts": rle_string.encode("utf-8"),
        }
        return coco_mask.decode(rle_dict).astype(np.uint8)

    # ------------------------------------------------------------------
    def _validate_and_resize(self, mask: np.ndarray) -> np.ndarray:
        """Ensure the mask is 2D and matches (height, width)."""
        if mask.ndim != 2:
            raise ValueError(
                f"Expected 2D binary mask, got shape {mask.shape}. "
                "Pass a single instance mask per call."
            )

        h, w = mask.shape
        if h != self.height or w != self.width:
            import cv2
            mask = cv2.resize(
                mask.astype(np.uint8),
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )

        return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Submission CSV builder
# ---------------------------------------------------------------------------

def build_submission_csv(
    predictions: Dict[str, List[np.ndarray]],
    height: int = 2048,
    width: int = 2048,
) -> pd.DataFrame:
    """
    Build the MAGFiLO benchmark submission CSV.

    Parameters
    ----------
    predictions : dict[str, list[np.ndarray]]
        Keys are image IDs (e.g. '20150125172714Mh').
        Values are lists of binary masks (H, W) uint8 — one per predicted
        filament instance.

    height, width : int
        Output image dimensions (fixed at 2048 × 2048 for MAGFiLO).

    Returns
    -------
    pd.DataFrame with columns: filament_id, segmentation_rle
        Ready to save with ``df.to_csv('submission.csv', index=False)``.

    Examples
    --------
    >>> preds = {
    ...     '20150125172714Mh': [mask1, mask2, mask3],
    ...     '20170501024112Bh': [mask4],
    ... }
    >>> df = build_submission_csv(preds)
    >>> df.to_csv('submission.csv', index=False)
    """
    encoder = RLEEncoder(height=height, width=width)
    rows: List[Dict[str, str]] = []

    for image_id, masks in predictions.items():
        if len(masks) == 0:
            # Submit an empty mask if no filaments were detected
            # (prevents the image from being skipped entirely)
            empty = np.zeros((height, width), dtype=np.uint8)
            rows.append({
                "filament_id": f"{image_id}_1",
                "segmentation_rle": encoder.encode(empty),
            })
            continue

        for instance_idx, mask in enumerate(masks, start=1):
            filament_id = f"{image_id}_{instance_idx}"
            rle_string = encoder.encode(mask)
            rows.append({
                "filament_id": filament_id,
                "segmentation_rle": rle_string,
            })

    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
