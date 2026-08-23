"""
Oriented Region Proposal Network (Oriented RPN).

Implements midpoint-offset based rotated bounding box regression to avoid
angular boundary discontinuities that occur when regressing θ directly.

Architecture
------------
1. ``OrientedAnchorGenerator``
   Creates a grid of anchors at each FPN level.  Each anchor is defined
   by (scale, ratio, angle_deg) and expressed as (xc, yc, w, h, θ_rad).

2. ``MidpointOffsetBoxCoder``
   Encodes/decodes oriented boxes using midpoint-offset representation
   Δ = (Δx, Δy, Δw, Δh, Δα, Δβ) — no angular discontinuities.

3. ``OrientedRPN``
   For each FPN level, applies a shared convolutional head predicting:
   - cls_logits : (B, A, H_l, W_l)  — objectness per anchor
   - box_deltas : (B, A*6, H_l, W_l) — 6 midpoint offsets per anchor

   During training, anchors are matched to GT boxes; losses are computed.
   During inference, top-k proposals are returned after rotated NMS.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opt_hq_net.config import OrientedRPNConfig


# ---------------------------------------------------------------------------
# Anchor generator
# ---------------------------------------------------------------------------

class OrientedAnchorGenerator:
    """
    Generate oriented anchors on a regular grid for each FPN feature level.

    Each (scale, ratio, angle) triple defines one anchor template.  Anchors
    are represented as (xc, yc, w, h, θ_rad) and centered on each spatial
    location of the feature map.

    Parameters
    ----------
    anchor_scales : list[int]
        Anchor sizes in pixels (before aspect-ratio adjustment).
    anchor_ratios : list[float]
        Aspect ratios w/h.  High ratios (e.g. 4, 8) capture elongated filaments.
    anchor_angles : list[float]
        Template orientations in degrees [0, 180).
    strides : list[int]
        Stride of each FPN level (e.g. [4, 8, 16, 32] for P2…P5).
    """

    def __init__(
        self,
        anchor_scales: List[int],
        anchor_ratios: List[float],
        anchor_angles: List[float],
        strides: List[int],
    ) -> None:
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_angles = [math.radians(a) for a in anchor_angles]
        self.strides = strides

        # Pre-compute anchor templates for the base cell (per level)
        self._templates = self._build_templates()
        self.num_anchors_per_location = len(self._templates)

    # ------------------------------------------------------------------
    def _build_templates(self) -> torch.Tensor:
        """Return anchor templates (A, 5) at a single grid cell origin."""
        templates = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                area = scale ** 2
                w = math.sqrt(area * ratio)
                h = math.sqrt(area / ratio)
                for theta in self.anchor_angles:
                    templates.append([0.0, 0.0, w, h, theta])
        return torch.tensor(templates, dtype=torch.float32)  # (A, 5)

    # ------------------------------------------------------------------
    def generate_anchors(
        self,
        feature_shapes: List[Tuple[int, int]],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Generate all anchors for all FPN levels.

        Parameters
        ----------
        feature_shapes : list[(H_l, W_l)]
            Spatial size of each feature map level.
        device : torch.device

        Returns
        -------
        list[Tensor]
            One tensor per level, shape (H_l * W_l * A, 5).
        """
        all_anchors = []
        templates = self._templates.to(device)

        for (h, w), stride in zip(feature_shapes, self.strides):
            # Grid centres
            shift_x = (torch.arange(0, w, device=device) + 0.5) * stride
            shift_y = (torch.arange(0, h, device=device) + 0.5) * stride
            gy, gx = torch.meshgrid(shift_y, shift_x, indexing="ij")  # (H, W)

            # (H*W, 1, 2) centres + (1, A, 5) templates → broadcast
            centres = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (HW, 2)
            anchors = templates.unsqueeze(0).expand(len(centres), -1, -1).clone()
            anchors[:, :, 0] += centres[:, 0:1]   # shift xc
            anchors[:, :, 1] += centres[:, 1:2]   # shift yc

            all_anchors.append(anchors.reshape(-1, 5))   # (HW*A, 5)

        return all_anchors


# ---------------------------------------------------------------------------
# Box coder — midpoint offset representation
# ---------------------------------------------------------------------------

class MidpointOffsetBoxCoder:
    """
    Encode / decode oriented bounding boxes via midpoint offsets.

    The midpoint representation (used in Oriented R-CNN) avoids the angular
    boundary problem that plagues direct θ regression:

        Δx   = (gt_xc - anchor_xc) / anchor_w
        Δy   = (gt_yc - anchor_yc) / anchor_h
        Δw   = log(gt_w / anchor_w)
        Δh   = log(gt_h / anchor_h)
        Δα   = (gt_α_x - anchor_xc) / anchor_w  (top midpoint x-offset)
        Δβ   = (gt_α_y - anchor_yc) / anchor_h  (top midpoint y-offset)

    where (gt_α_x, gt_α_y) is the midpoint of the top side of the GT box.

    References
    ----------
    Han et al., "Oriented R-CNN for Object Detection", ICCV 2021.
    """

    def __init__(self, weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)) -> None:
        self.weights = weights

    # ------------------------------------------------------------------
    def encode(
        self, anchors: torch.Tensor, gt_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regression targets Δ ∈ R^6 for each (anchor, gt_box) pair.

        Parameters
        ----------
        anchors  : (N, 5) — [xc, yc, w, h, θ]
        gt_boxes : (N, 5) — [xc, yc, w, h, θ]

        Returns
        -------
        deltas : (N, 6) — [Δx, Δy, Δw, Δh, Δα, Δβ]
        """
        wa, ha = anchors[:, 2].clamp(min=1e-4), anchors[:, 3].clamp(min=1e-4)
        xa, ya = anchors[:, 0], anchors[:, 1]
        xg, yg = gt_boxes[:, 0], gt_boxes[:, 1]
        wg, hg = gt_boxes[:, 2], gt_boxes[:, 3]
        theta_g = gt_boxes[:, 4]

        # Top midpoint of GT box
        alpha_x = xg + 0.5 * hg * torch.sin(theta_g)
        alpha_y = yg - 0.5 * hg * torch.cos(theta_g)

        dx = self.weights[0] * (xg - xa) / wa
        dy = self.weights[1] * (yg - ya) / ha
        dw = self.weights[2] * torch.log(wg / wa)
        dh = self.weights[3] * torch.log(hg / ha)
        da = self.weights[4] * (alpha_x - xa) / wa
        db = self.weights[5] * (alpha_y - ya) / ha

        return torch.stack([dx, dy, dw, dh, da, db], dim=-1)

    # ------------------------------------------------------------------
    def decode(
        self, anchors: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply regression deltas to anchors to get predicted oriented boxes.

        Parameters
        ----------
        anchors : (N, 5)
        deltas  : (N, 6)

        Returns
        -------
        boxes : (N, 5) — [xc, yc, w, h, θ]
        """
        wa, ha = anchors[:, 2], anchors[:, 3]
        xa, ya = anchors[:, 0], anchors[:, 1]

        dx = deltas[:, 0] / self.weights[0]
        dy = deltas[:, 1] / self.weights[1]
        dw = deltas[:, 2] / self.weights[2]
        dh = deltas[:, 3] / self.weights[3]
        da = deltas[:, 4] / self.weights[4]
        db = deltas[:, 5] / self.weights[5]

        xg = dx * wa + xa
        yg = dy * ha + ya
        wg = torch.exp(dw) * wa
        hg = torch.exp(dh) * ha

        # Reconstruct θ from top midpoint offsets
        alpha_x = da * wa + xa
        alpha_y = db * ha + ya
        theta_g = torch.atan2(alpha_x - xg, -(alpha_y - yg))

        return torch.stack([xg, yg, wg, hg, theta_g], dim=-1)


# ---------------------------------------------------------------------------
# Oriented RPN head
# ---------------------------------------------------------------------------

class OrientedRPN(nn.Module):
    """
    Oriented Region Proposal Network.

    Operates over all FPN levels simultaneously.  Uses a shared
    convolutional head (3×3 conv → two parallel 1×1 convs) at each level
    to predict objectness and midpoint offsets.

    Parameters
    ----------
    in_channels : int
        FPN feature map channel width (same for all levels).
    cfg : OrientedRPNConfig
        RPN configuration object.

    Training output
    ---------------
    dict with keys: 'rpn_cls_loss', 'rpn_box_loss'

    Inference output
    ----------------
    list[Tensor] — one (K, 5+1) tensor per image with proposed
    oriented boxes [xc, yc, w, h, θ, score].
    """

    def __init__(self, in_channels: int, cfg: OrientedRPNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Anchor generator (strides match FPN levels P2…P5)
        self.anchor_gen = OrientedAnchorGenerator(
            anchor_scales=cfg.anchor_scales,
            anchor_ratios=cfg.anchor_ratios,
            anchor_angles=cfg.anchor_angles,
            strides=[4, 8, 16, 32],
        )
        self.A = self.anchor_gen.num_anchors_per_location
        self.box_coder = MidpointOffsetBoxCoder()

        # Shared feature head (applied identically to every FPN level)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Classification head: A objectness logits per location
        self.cls_head = nn.Conv2d(in_channels, self.A, kernel_size=1)
        # Regression head: A * 6 midpoint deltas per location
        self.reg_head = nn.Conv2d(in_channels, self.A * 6, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for layer in [self.conv, self.cls_head, self.reg_head]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0)
        # Initialise cls bias for ~1% prior objectness
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))

    # ------------------------------------------------------------------
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        gt_boxes: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple:
        """
        Forward pass.

        Parameters
        ----------
        features : dict[str, Tensor]
            FPN feature maps {'P2': …, 'P3': …, 'P4': …, 'P5': …}.
        gt_boxes : list[Tensor] | None
            Ground-truth oriented boxes per image, each (N_i, 5).
            Required during training.
        image_sizes : list[(H, W)] | None
            Original image sizes for clipping proposals.

        Returns
        -------
        Training   : (proposals, loss_dict)
        Inference  : (proposals,)

        proposals : list[Tensor] — one per image, shape (K, 6): [xc,yc,w,h,θ,score]
        """
        level_names = ["P2", "P3", "P4", "P5"]
        level_features = [features[k] for k in level_names if k in features]

        cls_logits_all: List[torch.Tensor] = []
        box_deltas_all: List[torch.Tensor] = []

        for feat in level_features:
            x = self.conv(feat)
            cls_logits_all.append(self.cls_head(x))
            box_deltas_all.append(self.reg_head(x))

        # Decode proposals for each image
        feature_shapes = [f.shape[-2:] for f in level_features]
        device = level_features[0].device
        B = level_features[0].shape[0]

        all_anchors = self.anchor_gen.generate_anchors(feature_shapes, device)
        # Flatten all levels into a single anchor list per image
        anchors_flat = torch.cat(all_anchors, dim=0)  # (total_anchors, 5)

        proposals = self._decode_proposals(
            cls_logits_all, box_deltas_all, anchors_flat, feature_shapes, B
        )

        if self.training and gt_boxes is not None:
            losses = self._compute_losses(
                cls_logits_all, box_deltas_all, anchors_flat,
                gt_boxes, feature_shapes, B
            )
            return proposals, losses

        return proposals, {}

    # ------------------------------------------------------------------
    def _decode_proposals(
        self,
        cls_logits: List[torch.Tensor],
        box_deltas: List[torch.Tensor],
        anchors: torch.Tensor,
        feature_shapes: List[Tuple[int, int]],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """Decode top-K oriented proposals per image."""
        proposals_per_img = []
        A = self.A

        # Flatten logits and deltas across levels
        cls_flat_list, delta_flat_list = [], []
        for cls, delta in zip(cls_logits, box_deltas):
            B, _, H, W = cls.shape
            # cls: (B, A, H, W) → (B, H*W*A)
            cls_flat_list.append(cls.permute(0, 2, 3, 1).reshape(B, -1))
            # delta: (B, A*6, H, W) → (B, H*W*A, 6)
            delta_flat_list.append(
                delta.permute(0, 2, 3, 1).reshape(B, H * W * A, 6)
            )

        cls_flat = torch.cat(cls_flat_list, dim=1)      # (B, total_anchors)
        delta_flat = torch.cat(delta_flat_list, dim=1)  # (B, total_anchors, 6)

        scores = torch.sigmoid(cls_flat)

        for i in range(batch_size):
            boxes = self.box_coder.decode(anchors, delta_flat[i])  # (N, 5)
            s = scores[i]                                            # (N,)

            # Top-K by score
            k = min(self.cfg.pre_nms_top_n_test, len(s))
            top_idx = s.topk(k).indices
            boxes, s = boxes[top_idx], s[top_idx]

            # Threshold
            keep = s >= 0.05
            boxes, s = boxes[keep], s[keep]

            # Concatenate score as 6th column
            result = torch.cat([boxes, s.unsqueeze(-1)], dim=-1)
            proposals_per_img.append(result)

        return proposals_per_img

    # ------------------------------------------------------------------
    def _compute_losses(
        self,
        cls_logits: List[torch.Tensor],
        box_deltas: List[torch.Tensor],
        anchors: torch.Tensor,
        gt_boxes_list: List[torch.Tensor],
        feature_shapes: List[Tuple[int, int]],
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute RPN classification and box regression losses."""
        A = self.A
        cls_flat_list, delta_flat_list = [], []
        for cls, delta in zip(cls_logits, box_deltas):
            B, _, H, W = cls.shape
            cls_flat_list.append(cls.permute(0, 2, 3, 1).reshape(B, -1))
            delta_flat_list.append(
                delta.permute(0, 2, 3, 1).reshape(B, H * W * A, 6)
            )
        cls_flat = torch.cat(cls_flat_list, dim=1)
        delta_flat = torch.cat(delta_flat_list, dim=1)

        total_cls_loss = cls_flat.new_zeros(1)
        total_box_loss = cls_flat.new_zeros(1)

        for i in range(batch_size):
            gt = gt_boxes_list[i]      # (G, 5)
            logits = cls_flat[i]       # (N,)
            deltas = delta_flat[i]     # (N, 6)

            if len(gt) == 0:
                total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                    logits, torch.zeros_like(logits), reduction="mean"
                )
                continue

            # IoU matching (axis-aligned approximation for speed)
            iou = self._approx_iou(anchors, gt)  # (N, G)
            max_iou, gt_idx = iou.max(dim=1)

            labels = torch.zeros_like(logits)
            labels[max_iou >= self.cfg.fg_iou_threshold] = 1.0
            labels[max_iou < self.cfg.bg_iou_threshold] = 0.0
            # Ignore ambiguous
            ignore = (max_iou >= self.cfg.bg_iou_threshold) & (
                max_iou < self.cfg.fg_iou_threshold
            )

            pos_mask = labels == 1.0
            # Box loss on positives only
            if pos_mask.any():
                matched_gt = gt[gt_idx[pos_mask]]
                target_deltas = self.box_coder.encode(anchors[pos_mask], matched_gt)
                total_box_loss = total_box_loss + F.smooth_l1_loss(
                    deltas[pos_mask], target_deltas, reduction="mean"
                )

            valid = ~ignore
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                logits[valid], labels[valid], reduction="mean"
            )

        n = float(batch_size)
        return {
            "rpn_cls_loss": total_cls_loss / n,
            "rpn_box_loss": total_box_loss / n,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _approx_iou(anchors: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Approximate axis-aligned IoU between anchors and GT boxes for fast
        matching during training.  Uses the bounding rectangle of each OBB.
        """
        def to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
            xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            return torch.stack([
                xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            ], dim=1)

        a = to_xyxy(anchors)   # (N, 4)
        g = to_xyxy(gt)        # (G, 4)

        inter_x1 = torch.max(a[:, 0].unsqueeze(1), g[:, 0].unsqueeze(0))
        inter_y1 = torch.max(a[:, 1].unsqueeze(1), g[:, 1].unsqueeze(0))
        inter_x2 = torch.min(a[:, 2].unsqueeze(1), g[:, 2].unsqueeze(0))
        inter_y2 = torch.min(a[:, 3].unsqueeze(1), g[:, 3].unsqueeze(0))

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        union = area_a.unsqueeze(1) + area_g.unsqueeze(0) - inter_area

        return inter_area / union.clamp(min=1e-6)
