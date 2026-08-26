"""
Exponential Moving Average (EMA) for Filament-HQ Model Weights.

Stabilizes validation metrics and prevents high-frequency loss variance oscillations.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average of PyTorch model parameters.

    Parameters
    ----------
    model : nn.Module
        Base PyTorch model.
    decay : float
        EMA decay coefficient (default 0.999).
    device : str, optional
        Device to store EMA weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cuda") -> None:
        self.module = copy.deepcopy(model).eval().to(device)
        self.decay = decay
        self.device = device
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters using current model weights."""
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(self.decay * v + (1.0 - self.decay) * msd[k].to(self.device))

    @torch.no_grad()
    def set(self, model: nn.Module) -> None:
        """Hard reset EMA parameters to match current model weights."""
        self.module.load_state_dict(model.state_dict())
