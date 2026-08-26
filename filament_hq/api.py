"""
FilamentHQ Ultralytics-Style Top-Level API Wrapper.

Provides clean, intuitive framework methods:
  - model = FilamentHQ("convnext_tiny")
  - model.train(data_root="path/to/data", imgsz=1024, epochs=50, batch_size=2)
  - results = model.predict(source="path/to/image.png", imgsz=1024)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from filament_hq.data.dataset import FilamentTileDataset, filament_hq_collate_fn
from filament_hq.data.preprocessor import SolarPhysicalPreprocessor
from filament_hq.data.tiler import ImageTiler, TileStitcher
from filament_hq.engine.trainer import FilamentTrainer
from filament_hq.models.model import FilamentHQModel


class FilamentHQ:
    """
    Filament-HQ High-Resolution Segmentation Model.

    Parameters
    ----------
    backbone : str
        Backbone model architecture (default 'convnext_tiny').
    weights : str | Path, optional
        Path to pretrained model checkpoint weights.
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        version: str = "v2",
        weights: Optional[str | Path] = None,
    ) -> None:
        self.backbone = backbone
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FilamentHQModel(backbone=backbone, version=version, in_channels=4, embed_dim=16).to(self.device)
        self.preprocessor = SolarPhysicalPreprocessor()

        if weights and Path(weights).exists():
            print(f"[FilamentHQ] Loading pretrained weights from '{weights}'...")
            state_dict = torch.load(weights, map_location=self.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            # Flexible Warm-Start loading (load matching layers with automatic dtype alignment)
            model_sd = self.model.state_dict()
            matched_sd = {
                k: (v.to(model_sd[k].dtype) if torch.is_floating_point(model_sd[k]) else v)
                for k, v in state_dict.items()
                if k in model_sd and model_sd[k].shape == v.shape
            }
            model_sd.update(matched_sd)
            self.model.load_state_dict(model_sd)
            print(f"  [Warm-Start Success] Loaded {len(matched_sd)}/{len(model_sd)} matching layer tensors into FilamentHQ {version.upper()}!")

    def train(
        self,
        data_root: str | Path,
        imgsz: int = 1024,
        epochs: int = 50,
        batch_size: int = 2,
        lr: float = 1e-4,
        stage: int = 1,
        use_amp: bool = False,
        overfit_single_image: bool = False,
        checkpoint_dir: str = "checkpoints_hq",
        num_workers: int = 2,
        cache_dir: Optional[str | Path] = None,
    ) -> Dict[str, float]:
        """
        Train the FilamentHQ model on high-resolution dataset.

        Parameters
        ----------
        data_root : str | Path
            Root path to dataset folder containing images/ and masks/.
        imgsz : int
            Fixed input tile size (default 1024).
        epochs : int
            Number of training epochs (default 50).
        batch_size : int
            Batch size per GPU (default 2).
        lr : float
            Learning rate (default 1e-4).
        stage : int
            Curriculum loss stage (1, 2, or 3).
        use_amp : bool
            Enable Automatic Mixed Precision.
        overfit_single_image : bool
            Phase 0 overfit test mode (trains on 1 image only).
        cache_dir : str | Path, optional
            Optional directory for caching processed tiles.
        """
        # Create Train & Val Datasets
        train_ds = FilamentTileDataset(
            data_root=data_root,
            tile_size=imgsz,
            fg_prob=0.8,
            augment=True,
            overfit_single_image=overfit_single_image,
            cache_dir=cache_dir,
        )

        workers = 0 if overfit_single_image else num_workers

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(workers > 0),
            collate_fn=filament_hq_collate_fn,
        )

        val_ds = FilamentTileDataset(
            data_root=data_root,
            tile_size=imgsz,
            fg_prob=0.0,
            augment=False,
            overfit_single_image=overfit_single_image,
            cache_dir=cache_dir,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(workers > 0),
            collate_fn=filament_hq_collate_fn,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        trainer = FilamentTrainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=self.device,
            epochs=epochs,
            stage=stage,
            use_amp=use_amp,
            checkpoint_dir=checkpoint_dir,
            overfit_single_image=overfit_single_image,
        )

        return trainer.train()

    @torch.no_grad()
    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        imgsz: int = 1024,
        stride: int = 768,
        conf_thresh: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Run high-resolution tiled inference on a single 2048x2048 image.

        Returns
        -------
        dict with keys:
            'semantic' : (H, W) float32 probability map [0, 1]
            'boundary' : (H, W) float32 probability map [0, 1]
            'skeleton' : (H, W) float32 probability map [0, 1]
            'binary_mask' : (H, W) uint8 binary mask
        """
        self.model.eval()

        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        else:
            img = source.copy()

        # 1. Apply 4-channel physical preprocessor
        ch4_img = self.preprocessor(img)  # (H, W, 4)
        h, w = ch4_img.shape[:2]

        # 2. Extract 1024x1024 overlapping tiles
        tiler = ImageTiler(tile_size=imgsz, stride=stride)
        tiles, coords = tiler.extract_tiles(ch4_img)

        # 3. Setup Stitchers for each head
        sem_stitcher = TileStitcher(full_shape=(h, w), tile_size=imgsz, num_channels=1)
        bnd_stitcher = TileStitcher(full_shape=(h, w), tile_size=imgsz, num_channels=1)
        skl_stitcher = TileStitcher(full_shape=(h, w), tile_size=imgsz, num_channels=1)

        # 4. Predict tile outputs
        for tile, coord in zip(tiles, coords):
            # tile [1024, 1024, 4] -> tensor (1, 4, 1024, 1024)
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

            outputs = self.model(tile_tensor)
            sem_prob = torch.sigmoid(outputs["semantic"]).cpu().squeeze().numpy()
            bnd_prob = torch.sigmoid(outputs["boundary"]).cpu().squeeze().numpy()
            skl_prob = torch.sigmoid(outputs["skeleton"]).cpu().squeeze().numpy()

            sem_stitcher.add_tile(sem_prob, coord)
            bnd_stitcher.add_tile(bnd_prob, coord)
            skl_stitcher.add_tile(skl_prob, coord)

        # 5. Retrieve Gaussian-blended full-resolution maps
        full_sem = sem_stitcher.get_stitched_map()
        full_bnd = bnd_stitcher.get_stitched_map()
        full_skl = skl_stitcher.get_stitched_map()

        binary_mask = (full_sem > conf_thresh).astype(np.uint8)

        return {
            "semantic": full_sem,
            "boundary": full_bnd,
            "skeleton": full_skl,
            "binary_mask": binary_mask,
        }
