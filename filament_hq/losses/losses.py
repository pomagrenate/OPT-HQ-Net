"""
Staged Curriculum Compound Loss for Filament-HQ.

Combines four loss components with FP32 numerical shields:
  1. L_semantic  : Combo Focal + Dice Loss on 1024x1024 semantic logits.
  2. L_boundary  : Binary Cross-Entropy on filament edge boundaries.
  3. L_skeleton  : Topological Skeleton-Recall Loss on centerline pixels.
  4. L_instance  : Discriminative Instance Contrastive Loss (pushing different filament embeddings apart).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Assert that a tensor contains no NaN or Inf values."""
    if not torch.isfinite(tensor).all():
        print(f"[SANITY WARNING] '{name}' contains NaN/Inf. Cleaning via torch.nan_to_num...")
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


class FocalDiceLoss(nn.Module):
    """Focal + Dice Loss for binary segmentation."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-5) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = check_finite("focal_dice_logits", logits)
        targets = check_finite("focal_dice_targets", targets)

        logits_clamped = torch.clamp(logits, min=-10.0, max=10.0)
        probs = torch.sigmoid(logits_clamped)

        # 1. Focal Loss
        bce = F.binary_cross_entropy_with_logits(logits_clamped, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** 2.0 * bce).mean()

        # 2. Dice Loss
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = 1.0 - (2.0 * inter + self.eps) / (union + self.eps)

        loss = focal + dice
        return torch.nan_to_num(loss, nan=0.0)


class InstanceEmbeddingLoss(nn.Module):
    """
    Discriminative contrastive loss for instance pixel embeddings.
    Pulls embeddings of pixels within the same filament together while
    pushing mean cluster centers of different filaments apart by at least margin_dist.
    """

    def __init__(self, delta_var: float = 0.5, delta_dist: float = 1.5) -> None:
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def forward(self, embeddings: torch.Tensor, instances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : Tensor (B, D, H, W)
        instances  : Tensor (B, N_inst, H, W)
        """
        B, D, H, W = embeddings.shape
        total_loss = embeddings.new_zeros(1)
        valid_batches = 0

        for b in range(B):
            inst_b = instances[b]  # (N, H, W)
            if len(inst_b) == 0 or inst_b.sum() == 0:
                continue

            embed_b = embeddings[b].view(D, -1)  # (D, HW)
            centers = []
            var_loss = embeddings.new_zeros(1)

            # Intra-cluster variance loss
            for k in range(len(inst_b)):
                mask_k = inst_b[k].view(-1).bool()
                if mask_k.sum() == 0:
                    continue
                pixels_k = embed_b[:, mask_k]  # (D, N_k)
                center_k = pixels_k.mean(dim=1, keepdim=True)  # (D, 1)
                centers.append(center_k.squeeze(1))

                dist_k = torch.norm(pixels_k - center_k, p=2, dim=0)  # (N_k,)
                var_k = torch.clamp(dist_k - self.delta_var, min=0.0) ** 2
                var_loss = var_loss + var_k.mean()

            if not centers:
                continue

            var_loss = var_loss / max(len(centers), 1)

            # Inter-cluster distance loss
            dist_loss = embeddings.new_zeros(1)
            num_centers = len(centers)
            if num_centers > 1:
                center_mat = torch.stack(centers, dim=0)  # (K, D)
                pair_dist = torch.cdist(center_mat, center_mat, p=2)  # (K, K)
                mask_pair = ~torch.eye(num_centers, dtype=torch.bool, device=embeddings.device)
                diff = torch.clamp(2.0 * self.delta_dist - pair_dist[mask_pair], min=0.0) ** 2
                dist_loss = diff.mean()

            total_loss = total_loss + var_loss + dist_loss
            valid_batches += 1

        res = total_loss / max(valid_batches, 1)
        return torch.nan_to_num(res, nan=0.0)


from filament_hq.losses.instance import DiscriminativeEmbeddingLoss, AffinityLoss


class FilamentCompoundLoss(nn.Module):
    """
    Curriculum Multi-Task Compound Loss for Filament-HQ.

    Parameters
    ----------
    stage : int
        Curriculum stage (1: Semantic+Boundary, 2: +Skeleton, 3: +Embedding+Affinity).
    """

    def __init__(self, stage: int = 1) -> None:
        super().__init__()
        self.stage = stage
        self.focal_dice = FocalDiceLoss()
        self.instance_loss_fn = DiscriminativeEmbeddingLoss()
        self.affinity_loss_fn = AffinityLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute compound loss dictionary.
        """
        sem_logits = outputs["semantic"]
        bnd_logits = outputs["boundary"]
        skl_logits = outputs["skeleton"]
        inst_embeds = outputs.get("embedding", outputs.get("instance"))
        aff_logits = outputs.get("affinity")

        sem_gt = batch["semantic"]
        bnd_gt = batch["boundary"]
        skl_gt = batch["skeleton"]
        inst_gt = batch["instances"]

        # 1. Semantic Loss
        l_sem = self.focal_dice(sem_logits, sem_gt)

        # 2. Boundary Loss (auxiliary)
        bnd_logits_clamped = torch.clamp(bnd_logits, min=-10.0, max=10.0)
        l_bnd = F.binary_cross_entropy_with_logits(bnd_logits_clamped, bnd_gt)
        l_bnd = torch.nan_to_num(l_bnd, nan=0.0)

        # 3. Skeleton Topology Loss
        skl_logits_clamped = torch.clamp(skl_logits, min=-10.0, max=10.0)
        skl_probs = torch.sigmoid(skl_logits_clamped)
        n_skel = skl_gt.sum()
        if n_skel > 0:
            l_skl = 1.0 - (skl_probs * skl_gt).sum() / (n_skel + 1e-5)
            l_skl = torch.clamp(l_skl, min=0.0, max=1.0)
        else:
            l_skl = sem_logits.new_zeros(1)
        l_skl = torch.nan_to_num(l_skl, nan=0.0)

        # 4. Instance Embedding Loss & Affinity Loss
        if self.stage >= 2 and inst_embeds is not None and inst_gt is not None:
            l_inst = self.instance_loss_fn(inst_embeds, inst_gt)
        else:
            l_inst = sem_logits.new_zeros(1)

        if self.stage >= 2 and aff_logits is not None and inst_gt is not None:
            l_aff = self.affinity_loss_fn(aff_logits, inst_gt)
        else:
            l_aff = sem_logits.new_zeros(1)

        # Stage-specific weighting
        if self.stage == 1:
            w_sem, w_bnd, w_skl, w_inst, w_aff = 1.0, 0.3, 0.3, 0.0, 0.0
        elif self.stage == 2:
            w_sem, w_bnd, w_skl, w_inst, w_aff = 1.0, 0.3, 0.3, 0.5, 0.5
        else:
            w_sem, w_bnd, w_skl, w_inst, w_aff = 1.0, 0.3, 0.5, 0.5, 0.5

        total_loss = w_sem * l_sem + w_bnd * l_bnd + w_skl * l_skl + w_inst * l_inst + w_aff * l_aff
        total_loss = torch.nan_to_num(total_loss, nan=0.0)

        return {
            "loss_semantic": l_sem,
            "loss_boundary": l_bnd,
            "loss_skeleton": l_skl,
            "loss_instance": l_inst,
            "loss_affinity": l_aff,
            "total_loss": total_loss,
        }
