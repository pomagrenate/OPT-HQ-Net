"""
Instance Embedding & Affinity Loss Functions for Filament-HQ.

1. DiscriminativeEmbeddingLoss:
   Pulls pixel embeddings of the same filament toward their mean cluster center (variance penalty),
   and pushes cluster centers of different filaments apart (distance penalty).

2. AffinityLoss:
   Supervises horizontal (Ah) and vertical (Av) neighbor connectivity predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeEmbeddingLoss(nn.Module):
    """
    De Brabandere et al. Discriminative Loss Function for Instance Segmentation.

    Parameters
    ----------
    delta_var : float
        Variance margin (default 0.5).
    delta_dist : float
        Distance margin (default 1.5).
    norm : int
        Norm order for distance calculation (1 or 2).
    """

    def __init__(self, delta_var: float = 0.5, delta_dist: float = 1.5, norm: int = 2) -> None:
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm

    def forward(self, embeddings: torch.Tensor, instances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, C, H, W) float32
            16-D pixel embeddings.
        instances : (B, N, H, W) float32 / uint8
            Ground-truth instance masks.

        Returns
        -------
        loss : torch.Tensor
            Combined variance + distance loss.
        """
        b, c, h, w = embeddings.shape
        total_loss = embeddings.new_zeros(())
        valid_batches = 0

        for i in range(b):
            inst_masks = instances[i]  # (N, H, W)
            if inst_masks.ndim == 2:
                inst_masks = inst_masks.unsqueeze(0)

            # Filter non-empty masks
            non_empty = [m for m in inst_masks if m.sum() > 10]
            num_inst = len(non_empty)

            if num_inst == 0:
                continue

            valid_batches += 1
            emb_i = embeddings[i]  # (C, H, W)

            centers = []
            var_loss = embeddings.new_zeros(())

            # 1. Variance Loss: pull pixels to cluster center
            for mask in non_empty:
                m_bool = mask > 0
                pts_emb = emb_i[:, m_bool]  # (C, K)
                center = pts_emb.mean(dim=1, keepdim=True)  # (C, 1)
                centers.append(center.squeeze(1))

                dist = torch.norm(pts_emb - center, p=self.norm, dim=0) - self.delta_var
                var_loss = var_loss + F.relu(dist).pow(2).mean()

            var_loss = var_loss / num_inst

            # 2. Distance Loss: push cluster centers apart
            dist_loss = embeddings.new_zeros(())
            if num_inst > 1:
                centers_tensor = torch.stack(centers, dim=0)  # (N, C)
                for j in range(num_inst):
                    for k in range(j + 1, num_inst):
                        c_dist = torch.norm(centers_tensor[j] - centers_tensor[k], p=self.norm)
                        dist_penalty = 2.0 * self.delta_dist - c_dist
                        dist_loss = dist_loss + F.relu(dist_penalty).pow(2)
                dist_loss = dist_loss / (num_inst * (num_inst - 1) / 2.0)

            total_loss = total_loss + (var_loss + dist_loss)

        return total_loss / max(valid_batches, 1)


class AffinityLoss(nn.Module):
    """
    Horizontal & Vertical Pixel Instance Affinity Loss.

    Supervises neighbor connectivity:
      - Ah(x, y) = 1 if (x, y) and (x, y+1) belong to the same instance.
      - Av(x, y) = 1 if (x, y) and (x+1, y) belong to the same instance.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_affinity: torch.Tensor, instances: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_affinity : (B, 2, H, W) float32
            Predicted horizontal (ch0) and vertical (ch1) affinity logits.
        instances : (B, N, H, W) float32
            Instance masks.

        Returns
        -------
        loss : torch.Tensor
        """
        b, _, h, w = pred_affinity.shape
        device = pred_affinity.device

        # Compute GT Affinities
        gt_ah = torch.zeros((b, h, w), device=device)
        gt_av = torch.zeros((b, h, w), device=device)

        for i in range(b):
            inst_masks = instances[i]  # (N, H, W)
            if inst_masks.ndim == 2 or len(inst_masks) == 0:
                continue

            # Build labeled instance map
            inst_map = torch.zeros((h, w), device=device, dtype=torch.long)
            for idx, mask in enumerate(inst_masks, start=1):
                inst_map[mask > 0] = idx

            # Horizontal affinity (x, y) == (x, y+1)
            same_h = (inst_map[:, :-1] == inst_map[:, 1:]) & (inst_map[:, :-1] > 0)
            gt_ah[i, :, :-1] = same_h.float()

            # Vertical affinity (x, y) == (x+1, y)
            same_v = (inst_map[:-1, :] == inst_map[1:, :]) & (inst_map[:-1, :] > 0)
            gt_av[i, :-1, :] = same_v.float()

        loss_h = self.bce(pred_affinity[:, 0], gt_ah)
        loss_v = self.bce(pred_affinity[:, 1], gt_av)

        return (loss_h + loss_v) * 0.5
