"""
OPT-HQ Net: Full model assembly using the Builder Pattern.

End-to-end forward pass:
    image (B,3,H,W)
        │
        ▼ BackboneWithFPN
    {P2, P3, P4, P5}
        │
        ▼ OrientedRPN
    proposals  list[(K_i, 6)]  [xc,yc,w,h,θ,score]
        │
        ▼ MultiScaleRotatedRoIAlign
    roi_crops  (N_total, C, 28, 28)
        │
        ▼ HQMaskDecoder (fuses P2)
    mask_logits  (N_total, 1, 112, 112)
        │
        ▼  sigmoid + upsample to target resolution
    predictions  list[{'boxes': …, 'masks': …, 'scores': …}]

Training additionally returns a ``loss_dict`` with all component losses.

Usage (Builder Pattern)
-----------------------
>>> from opt_hq_net import OPTHQNetBuilder, ModelConfig
>>> cfg = ModelConfig(backbone_name="convnext_large", backbone_pretrained=True)
>>> model = OPTHQNetBuilder(cfg).build()
>>> model.train()
>>> output, losses = model(images, gt_boxes=gt_boxes, gt_masks=gt_masks)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_hq_net.config import ModelConfig
from opt_hq_net.losses.combined_loss import OPTHQNetLoss
from opt_hq_net.models.backbone import BackboneFactory
from opt_hq_net.models.mask_decoder import HQMaskDecoder
from opt_hq_net.models.oriented_rpn import OrientedRPN
from opt_hq_net.models.rotated_roi_align import MultiScaleRotatedRoIAlign


# ---------------------------------------------------------------------------
# OPT-HQ Net
# ---------------------------------------------------------------------------

class OPTHQNet(nn.Module):
    """
    Oriented-Prompted Topological High-Quality Network.

    Parameters
    ----------
    backbone : nn.Module
        BackboneWithFPN instance (output: dict P2…P5).
    rpn : OrientedRPN
        Oriented Region Proposal Network.
    roi_align : MultiScaleRotatedRoIAlign
        Multi-scale rotated crop extractor.
    decoder : HQMaskDecoder
        High-quality mask decoder.
    loss_fn : OPTHQNetLoss
        Combined multi-task loss (Oriented Box + Focal + Dice + Skeleton).
    target_mask_size : int
        Final mask upscale target (default 512 for memory efficiency;
        use 2048 for full-resolution submission).
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: OrientedRPN,
        roi_align: MultiScaleRotatedRoIAlign,
        decoder: HQMaskDecoder,
        loss_fn: OPTHQNetLoss,
        target_mask_size: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_align = roi_align
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.target_mask_size = target_mask_size

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[List[torch.Tensor]] = None,
        gt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple:
        """
        Forward pass.

        Parameters
        ----------
        images : Tensor (B, 3, H, W)
            Normalised input images.
        gt_boxes : list[Tensor (N_i, 5)] | None
            Ground-truth oriented boxes per image.  Required in train mode.
        gt_masks : list[Tensor (N_i, H, W)] | None
            Ground-truth binary masks per image.  Required in train mode.

        Returns
        -------
        Training:
            predictions : list[dict]   — per-image output dicts
            loss_dict   : dict[str, Tensor]  — all named losses + 'total_loss'

        Inference:
            predictions : list[dict]   — per-image output dicts
        """
        # ── 1. Feature extraction ──────────────────────────────────────
        features: Dict[str, torch.Tensor] = self.backbone(images)

        # ── 2. Oriented RPN ────────────────────────────────────────────
        image_sizes = [images.shape[-2:]] * images.shape[0]

        if self.training and gt_boxes is not None:
            proposals, rpn_losses = self.rpn(features, gt_boxes, image_sizes)
        else:
            proposals, rpn_losses = self.rpn(features, image_sizes=image_sizes)

        # ── 3. Rotated RoI Align ───────────────────────────────────────
        roi_crops, batch_idx = self.roi_align(features, proposals)

        # ── 4. HQ Mask Decoder ─────────────────────────────────────────
        p2 = features["P2"]
        mask_logits = self.decoder(roi_crops, proposals, p2, batch_idx)
        # mask_logits: (N_total, 1, 112, 112)

        # Upsample to target resolution during inference/evaluation only
        if not self.training and mask_logits.shape[0] > 0 and self.target_mask_size != 112:
            mask_logits = F.interpolate(
                mask_logits,
                size=(self.target_mask_size, self.target_mask_size),
                mode="bilinear",
                align_corners=False,
            )

        # ── 5. Compute losses (training only) ─────────────────────────
        if self.training and gt_boxes is not None and gt_masks is not None:
            mask_losses = self.loss_fn(
                mask_logits=mask_logits,
                proposals=proposals,
                batch_idx=batch_idx,
                gt_boxes=gt_boxes,
                gt_masks=gt_masks,
                image_size=(images.shape[-2], images.shape[-1]),
            )
            loss_dict = {**rpn_losses, **mask_losses}
            loss_dict["total_loss"] = sum(loss_dict.values())

            # Ensure all loss tensors are at least 1D (1,) so DataParallel can gather them across GPUs
            loss_dict = {
                k: (v.unsqueeze(0) if (isinstance(v, torch.Tensor) and v.ndim == 0) else v)
                for k, v in loss_dict.items()
            }
            return [], loss_dict

        # ── 6. Package per-image predictions (inference only) ─────────
        predictions = self._assemble_predictions(proposals, mask_logits, batch_idx, images.shape[0])
        return predictions, {}

    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_predictions(
        proposals: List[torch.Tensor],
        mask_logits: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
    ) -> List[Dict]:
        """
        Group mask logits and proposals back into per-image result dicts.

        Returns
        -------
        list of dicts, each with:
            'boxes'  : Tensor (N_i, 5) — oriented boxes [xc,yc,w,h,θ]
            'scores' : Tensor (N_i,)
            'masks'  : Tensor (N_i, H_out, W_out) — binary masks (sigmoid applied)
        """
        results = []
        for i in range(batch_size):
            idx = batch_idx == i
            if not idx.any():
                results.append({
                    "boxes": torch.zeros(0, 5),
                    "scores": torch.zeros(0),
                    "masks": torch.zeros(0, 1, 1),
                })
                continue

            prop = proposals[i]    # (N_i, 6)
            logits_i = mask_logits[idx]  # (N_i, 1, H, W)
            masks_prob = torch.sigmoid(logits_i.squeeze(1))  # (N_i, H, W)

            results.append({
                "boxes": prop[:, :5],
                "scores": prop[:, 5] if prop.shape[1] > 5 else torch.ones(len(prop)),
                "masks": masks_prob,
            })

        return results


# ---------------------------------------------------------------------------
# Builder Pattern
# ---------------------------------------------------------------------------

class OPTHQNetBuilder:
    """
    Builder for ``OPTHQNet``.

    Reads a ``ModelConfig`` and assembles all sub-modules with consistent
    channel widths, ensuring that FPN output dimensions flow correctly into
    the RPN, RoIAlign, and Decoder.

    Usage
    -----
    >>> cfg = ModelConfig()
    >>> model = OPTHQNetBuilder(cfg).build()
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    def build(self) -> OPTHQNet:
        """Assemble and return a fully-initialised ``OPTHQNet``."""
        cfg = self.cfg

        # 1. Backbone + FPN
        backbone = BackboneFactory.build(
            name=cfg.backbone_name,
            pretrained=cfg.backbone_pretrained,
            out_channels=cfg.fpn.out_channels,
        )

        # 2. Oriented RPN
        rpn = OrientedRPN(
            in_channels=cfg.fpn.out_channels,
            cfg=cfg.rpn,
        )

        # 3. Multi-scale Rotated RoIAlign
        roi_align = MultiScaleRotatedRoIAlign(
            output_size=cfg.roi_align.output_size,
        )

        # 4. HQ Mask Decoder
        # Ensure decoder p2_channels matches FPN out_channels
        cfg.decoder.p2_channels = cfg.fpn.out_channels
        decoder = HQMaskDecoder(cfg=cfg.decoder)

        # 5. Loss function (imported lazily to avoid circular imports)
        from opt_hq_net.config import LossWeightConfig
        loss_fn = OPTHQNetLoss()

        return OPTHQNet(
            backbone=backbone,
            rpn=rpn,
            roi_align=roi_align,
            decoder=decoder,
            loss_fn=loss_fn,
        )
