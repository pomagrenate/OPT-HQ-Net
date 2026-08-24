"""
MAGFiLO Offline Dataset Preprocessing Module inside opt_hq_net.data package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm


def masks_to_oriented_boxes(masks: np.ndarray) -> np.ndarray:
    """Compute oriented bounding boxes [xc, yc, w, h, theta_rad] from binary masks."""
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0, 0.0])
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        (xc, yc), (w, h), angle_deg = cv2.minAreaRect(pts)
        theta_rad = np.deg2rad(angle_deg)
        boxes.append([float(xc), float(yc), float(w), float(h), float(theta_rad)])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


def preprocess_magfilo_dataset(
    data_root: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: int = 512,
    show_progress: bool = True,
) -> Path:
    """
    Preprocess raw MAGFiLO dataset into downsampled images and compressed NPZ mask files.

    Parameters
    ----------
    data_root : str | Path
        Path to raw split directory containing 'train_images/' and/or COCO JSON.
    output_dir : str | Path
        Target directory to save preprocessed 'images/' and 'masks/' NPZ files.
    target_size : int, optional
        Target spatial size (512 or 768). Default is 512.
    show_progress : bool, optional
        Whether to display tqdm progress bar. Default is True.

    Returns
    -------
    Path
        Path object pointing to output_dir.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    img_out_dir = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate COCO JSON
    json_files = [j for j in (list(data_root.glob("*.json")) + list(data_root.rglob("*.json"))) if j.name.lower() != "manifest.json"]
    split_name = data_root.name.lower()
    matching_jsons = [j for j in json_files if split_name in j.name.lower() or split_name in j.parent.name.lower()]
    coco_json = matching_jsons[0] if matching_jsons else (json_files[0] if json_files else None)

    img_id_to_anns = {}
    if coco_json and coco_json.exists():
        print(f"[Preprocessor] Parsing COCO annotations from: {coco_json}")
        with open(coco_json, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        img_id_map = {
            str(img["id"]): (Path(img["file_name"]).stem, img.get("height", 2048), img.get("width", 2048))
            for img in coco_data.get("images", [])
        }
        for ann in coco_data.get("annotations", []):
            coco_img_id = str(ann["image_id"])
            if coco_img_id in img_id_map:
                stem, h, w = img_id_map[coco_img_id]
                img_id_to_anns.setdefault(stem, []).append({**ann, "_h": h, "_w": w})
        print(f"[Preprocessor] Loaded annotations for {len(img_id_to_anns)} image stems.")

    # 2. Locate image files
    img_dir = (
        data_root / "train_images"
        if (data_root / "train_images").exists()
        else (data_root / "images" if (data_root / "images").exists() else data_root)
    )
    img_paths = list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    print(f"[Preprocessor] Processing {len(img_paths)} images to resolution {target_size}x{target_size}...")

    manifest = []
    pbar = tqdm(img_paths, desc="Preprocessing", disable=not show_progress)
    for img_path in pbar:
        stem = img_path.stem
        # Read raw image
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            continue

        orig_h, orig_w = raw_img.shape[:2]

        # Resize image
        resized_img = cv2.resize(raw_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_out_dir / f"{stem}.jpeg"), resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Render masks at target resolution
        anns = img_id_to_anns.get(stem, [])
        masks_list = []
        if anns:
            scale_x = target_size / float(orig_w)
            scale_y = target_size / float(orig_h)
            for ann in anns:
                seg = ann.get("segmentation")
                mask = np.zeros((target_size, target_size), dtype=np.uint8)
                if isinstance(seg, list):
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        pts[:, 0] *= scale_x
                        pts[:, 1] *= scale_y
                        pts_int = pts.astype(np.int32)
                        cv2.fillPoly(mask, [pts_int], 1)
                masks_list.append(mask)

        if masks_list:
            masks_arr = np.stack(masks_list, axis=0)  # (N, target_size, target_size) uint8
            boxes_arr = masks_to_oriented_boxes(masks_arr)
        else:
            masks_arr = np.zeros((0, target_size, target_size), dtype=np.uint8)
            boxes_arr = np.zeros((0, 5), dtype=np.float32)

        # Save NPZ file
        np.savez_compressed(
            mask_out_dir / f"{stem}.npz",
            masks=masks_arr,
            boxes=boxes_arr,
        )

        manifest.append({
            "stem": stem,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "target_size": target_size,
            "num_instances": len(masks_arr),
        })

    # Save manifest
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Preprocessor] Complete! Preprocessed {len(manifest)} items saved to: {output_dir}")
    return output_dir
