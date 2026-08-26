"""
Multi-scale feature backbone with Feature Pyramid Network (FPN) neck.

Design: Strategy Pattern via ``BackboneFactory``.
    - Call ``BackboneFactory.build(name, ...)`` to get a ``BackboneWithFPN``.
    - Supported names: any timm model whose forward produces a feature list
      (ConvNeXt family, Swin Transformer family).

Output
------
A dict ``{'P2': Tensor, 'P3': Tensor, 'P4': Tensor, 'P5': Tensor}``
where Pi has stride 2^i relative to the input image and ``out_channels``
feature channels.

P2 = 1/4  scale  (high spatial detail — used by HQ token)
P3 = 1/8  scale
P4 = 1/16 scale
P5 = 1/32 scale  (global solar context)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lateral + top-down FPN implementation (no dependency on torchvision)
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """
    Lightweight Feature Pyramid Network neck.

    Converts a list of backbone feature maps with varying channels into
    a fixed ``out_channels`` pyramid via 1×1 lateral convolutions and
    top-down upsampling.

    Parameters
    ----------
    in_channels_list : list[int]
        Number of channels for each input feature map (bottom to top).
    out_channels : int
        Unified channel width across all pyramid levels.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
             for _ in in_channels_list]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        features : list[Tensor]
            Backbone feature maps ordered from shallowest (high-res) to
            deepest (low-res).  Length must equal ``len(in_channels_list)``.

        Returns
        -------
        dict[str, Tensor]
            Keys 'P2' … 'P{2+n}' with shape (B, out_channels, H/2^i, W/2^i).
        """
        # Step 1: lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Step 2: top-down pathway (from deepest to shallowest)
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest",
            )
            laterals[i] = laterals[i] + up

        # Step 3: output convolutions
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Return as named dict P2, P3, P4, P5 …
        return {f"P{i + 2}": out for i, out in enumerate(outs)}


# ---------------------------------------------------------------------------
# Dynamic Forward-Hook Feature Extractor (Fallback for SegFormer / ViT)
# ---------------------------------------------------------------------------

def enable_hf_segformer_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing natively for HuggingFace SegFormer models."""
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing = True
    if hasattr(model, "encoder"):
        model.encoder.gradient_checkpointing = True
    for m in model.modules():
        setattr(m, "gradient_checkpointing", True)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass


class HookedFeatureExtractor(nn.Module):
    """
    Generic feature extractor wrapper using PyTorch forward hooks or HuggingFace SegFormer.
    Extracts 4 multi-scale feature maps from any timm or HuggingFace model (e.g. SegFormer / MixTransformer)
    that lacks native ``features_only=True`` support in timm.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.captured_features: List[torch.Tensor] = []
        self.hooks = []
        self.is_hf_segformer = False

        # Attempt native HuggingFace SegFormer loading if SegFormer/mit is requested
        if any(term in model_name.lower() for term in ["segformer", "mit", "mix_transformer"]):
            try:
                from transformers import SegformerModel
                b_num = "2"
                for sub in ["b0", "b1", "b2", "b3", "b4", "b5"]:
                    if sub in model_name.lower():
                        b_num = sub[-1]
                        break
                hf_id = f"nvidia/mit-b{b_num}"
                print(f"[HookedFeatureExtractor] Loading HuggingFace SegFormer model: '{hf_id}'...")
                self.model = SegformerModel.from_pretrained(hf_id)
                enable_hf_segformer_checkpointing(self.model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[HookedFeatureExtractor] Enabled HuggingFace Gradient Checkpointing (saves ~65% VRAM).")
                self.is_hf_segformer = True
                return
            except Exception as hf_err:
                print(f"[HookedFeatureExtractor WARNING] HuggingFace SegFormer failed: {hf_err}. Trying timm hooked extraction...")

        self.model = timm.create_model(model_name, pretrained=pretrained)

        named_modules = dict(self.model.named_modules())
        target_names = []

        possible_patterns = [
            ["norm1", "norm2", "norm3", "norm4"],
            ["block1", "block2", "block3", "block4"],
            ["stage1", "stage2", "stage3", "stage4"],
            ["stages.0", "stages.1", "stages.2", "stages.3"],
            ["blocks.0", "blocks.1", "blocks.2", "blocks.3"],
        ]
        for pattern in possible_patterns:
            if all(p in named_modules for p in pattern):
                target_names = pattern
                break

        if not target_names:
            stage_candidates = [
                k for k, v in named_modules.items()
                if any(term in k.lower() for term in ["norm", "block", "stage", "layer"])
                and not "." in k
            ]
            if len(stage_candidates) >= 4:
                step = max(1, len(stage_candidates) // 4)
                target_names = [stage_candidates[i * step] for i in range(4)]

        if not target_names:
            all_keys = [k for k in named_modules.keys() if k]
            step = max(1, len(all_keys) // 4)
            target_names = [all_keys[i * step] for i in range(1, 5) if i * step < len(all_keys)]

        self.target_names = target_names
        for name in self.target_names:
            mod = named_modules[name]
            self.hooks.append(mod.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook(module, input, output):
            val = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(val, torch.Tensor):
                self.captured_features.append(val)
        return hook

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.is_hf_segformer:
            outputs = self.model(pixel_values=x, output_hidden_states=True)
            hs = list(outputs.hidden_states)
            # HF SegformerModel returns (embedding_output, stage1, stage2, stage3, stage4)
            if len(hs) == 5:
                return hs[1:]
            return hs

        self.captured_features = []
        try:
            _ = self.model(x)
        except Exception:
            pass
        return self.captured_features


# ---------------------------------------------------------------------------
# Backbone wrapper — feature extraction via timm
# ---------------------------------------------------------------------------

class BackboneWithFPN(nn.Module):
    """
    timm backbone paired with an FPN neck.

    Parameters
    ----------
    model_name : str
        timm model identifier, e.g. ``'convnext_large'`` or
        ``'swin_large_patch4_window7_224'``.
    pretrained : bool
        Load ImageNet pretrained weights.
    out_channels : int
        FPN output channel width (all levels share this width).
    out_indices : tuple[int]
        Which backbone stages to extract (0 = shallowest).  Default
        (1, 2, 3, 4) maps to P2 … P5 for ConvNeXt / Swin.

    Raises
    ------
    ImportError
        If ``timm`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "convnext_large",
        pretrained: bool = True,
        out_channels: int = 256,
        out_indices: tuple = (0, 1, 2, 3),
    ) -> None:
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for the backbone. "
                "Install with: pip install timm"
            )

        # Build kwargs for timm.create_model
        create_kwargs = {
            "pretrained": pretrained,
            "features_only": True,
            "out_indices": out_indices,
        }

        if "swin" in model_name.lower() or "vit" in model_name.lower():
            create_kwargs["strict_img_size"] = False
            create_kwargs["dynamic_img_pad"] = True

        # Create feature extractor (returns list of feature maps)
        self.is_hooked = False
        try:
            self.backbone = timm.create_model(model_name, **create_kwargs)
            self.in_channels_list: List[int] = self.backbone.feature_info.channels()
        except Exception as err:
            # Fallback: create base model and attach forward hooks (enables native SegFormer support)
            print(f"[Backbone] 'features_only=True' not natively supported for '{model_name}' ({err}). Enabling Hooked Feature Extraction...")
            self.backbone = HookedFeatureExtractor(model_name, pretrained=pretrained)
            self.is_hooked = True
            
            # Infer feature channels via a single dummy pass
            with torch.no_grad():
                dummy_x = torch.zeros(1, 3, 224, 224)
                dummy_feats = self.backbone(dummy_x)
                self.in_channels_list = []
                for f in dummy_feats:
                    if f.ndim == 4:
                        # NCHW vs NHWC channel check
                        c = f.shape[1] if f.shape[1] <= f.shape[-1] else f.shape[-1]
                        self.in_channels_list.append(c)
                    elif f.ndim == 3:
                        self.in_channels_list.append(f.shape[-1])
                if not self.in_channels_list:
                    raise RuntimeError(f"HookedFeatureExtractor failed to capture feature maps for model '{model_name}'.")

        # Enable gradient checkpointing to save up to 60% activation VRAM
        if hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
                print("[Backbone] Enabled Gradient Checkpointing (saves ~60% VRAM during backward pass)")
            except Exception as e:
                pass
        elif hasattr(getattr(self.backbone, "model", None), "set_grad_checkpointing"):
            try:
                self.backbone.model.set_grad_checkpointing(True)
                print("[Backbone] Enabled Gradient Checkpointing on underlying model.")
            except Exception:
                pass
        elif hasattr(getattr(self.backbone, "model", None), "gradient_checkpointing_enable"):
            try:
                self.backbone.model.gradient_checkpointing_enable()
                print("[Backbone] Enabled HuggingFace Gradient Checkpointing on underlying SegFormer model.")
            except Exception:
                pass

        self.fpn = FPNNeck(self.in_channels_list, out_channels)
        self.out_channels = out_channels

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing on the underlying timm backbone."""
        target = self.backbone if not self.is_hooked else getattr(self.backbone, "model", self.backbone)
        if hasattr(target, "set_grad_checkpointing"):
            try:
                target.set_grad_checkpointing(enable)
                print(f"[Backbone] Gradient Checkpointing set to: {enable}")
            except Exception as e:
                print(f"[Backbone WARNING] Failed to set gradient checkpointing: {e}")
        elif hasattr(target, "gradient_checkpointing_enable"):
            try:
                if enable:
                    enable_hf_segformer_checkpointing(target)
                    print(f"[Backbone] HuggingFace Gradient Checkpointing ENABLED.")
                else:
                    target.gradient_checkpointing_disable()
                    print(f"[Backbone] HuggingFace Gradient Checkpointing DISABLED.")
            except Exception as e:
                print(f"[Backbone WARNING] Failed to set HF gradient checkpointing: {e}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input image batch, shape (B, 3, H, W).

        Returns
        -------
        dict[str, Tensor]
            ``{'P2': ..., 'P3': ..., 'P4': ..., 'P5': ...}``
        """
        features: List[torch.Tensor] = self.backbone(x)

        # Swin Transformer, SegFormer, and ViT models in timm may output NHWC or token sequence tensors.
        # Permute/reshape to NCHW so FPN 2D convolutions receive channels in dim 1.
        formatted_features = []
        for i, feat in enumerate(features):
            if feat.ndim == 3:
                # Reshape (B, L, C) -> (B, C, H, W) assuming square spatial grid
                b, l, c = feat.shape
                h = w = int(l ** 0.5)
                if h * w == l:
                    feat = feat.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
            elif feat.ndim == 4:
                expected_c = self.in_channels_list[i] if i < len(self.in_channels_list) else None
                if expected_c is not None and feat.shape[1] != expected_c and feat.shape[-1] == expected_c:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                elif feat.shape[-1] < feat.shape[1] and feat.shape[-1] < feat.shape[2]:
                    feat = feat.permute(0, 3, 1, 2).contiguous()
            formatted_features.append(feat)

        return self.fpn(formatted_features)


# ---------------------------------------------------------------------------
# Strategy: BackboneFactory
# ---------------------------------------------------------------------------

class BackboneFactory:
    """
    Factory for creating backbone + FPN instances from a config string.

    Supported names (case-insensitive)
    -----------------------------------
    - ``'convnext_large'``      — ConvNeXt-Large (GELU, hierarchical CNN)
    - ``'swin_large'``          — Swin Transformer Large (window attention)
    - Any valid timm model name with ``features_only=True`` support.

    Examples
    --------
    >>> backbone = BackboneFactory.build("convnext_large", pretrained=True)
    >>> feats = backbone(torch.randn(1, 3, 1024, 1024))
    >>> print({k: v.shape for k, v in feats.items()})
    """

    # Alias map for user-friendly short names
    _ALIASES: Dict[str, str] = {
        "convnext_tiny": "convnext_tiny",
        "convnext-tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext-small": "convnext_small",
        "convnext_base": "convnext_base",
        "convnext-base": "convnext_base",
        "convnext_large": "convnext_large",
        "convnext-large": "convnext_large",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin-tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "swin-small": "swin_small_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
        "swin-base": "swin_base_patch4_window7_224",
        "swin_large": "swin_large_patch4_window7_224",
        "swin-large": "swin_large_patch4_window7_224",
        # SegFormer (MixTransformer) backbones in timm
        "segformer_b0": "mix_transformer_b0",
        "segformer-b0": "mix_transformer_b0",
        "segformer_b1": "mix_transformer_b1",
        "segformer-b1": "mix_transformer_b1",
        "segformer_b2": "mix_transformer_b2",
        "segformer-b2": "mix_transformer_b2",
        "segformer_b3": "mix_transformer_b3",
        "segformer-b3": "mix_transformer_b3",
        "segformer_b4": "mix_transformer_b4",
        "segformer-b4": "mix_transformer_b4",
        "segformer_b5": "mix_transformer_b5",
        "segformer-b5": "mix_transformer_b5",
        "mit_b0": "mix_transformer_b0",
        "mit_b1": "mix_transformer_b1",
        "mit_b2": "mix_transformer_b2",
        "mit_b3": "mix_transformer_b3",
        "mit_b4": "mix_transformer_b4",
        "mit_b5": "mix_transformer_b5",
        # HRNet backbones
        "hrnet_w18": "hrnet_w18",
        "hrnet-w18": "hrnet_w18",
        "hrnet_w32": "hrnet_w32",
        "hrnet-w32": "hrnet_w32",
        "hrnet_w48": "hrnet_w48",
        "hrnet-w48": "hrnet_w48",
    }

    @classmethod
    def build(
        cls,
        name: str,
        pretrained: bool = True,
        out_channels: int = 256,
        out_indices: tuple = (0, 1, 2, 3),
    ) -> BackboneWithFPN:
        """
        Build and return a ``BackboneWithFPN`` instance.

        Parameters
        ----------
        name : str
            Backbone name (see class docstring).
        pretrained : bool
            Load pretrained ImageNet weights.
        out_channels : int
            FPN output channel width.
        out_indices : tuple[int]
            Backbone stages to extract.
        """
        resolved = cls._ALIASES.get(name.lower(), name)
        
        # Multi-candidate list to handle variations across timm versions
        candidates = [resolved]
        if "segformer" in resolved or "mit" in resolved or "mix_transformer" in resolved:
            b_num = name.lower().split("b")[-1] if "b" in name.lower() else "2"
            candidates.extend([
                f"mix_transformer_b{b_num}",
                f"mit_b{b_num}",
                f"segformer_b{b_num}",
                f"hf_hub:nvidia/mit-b{b_num}",
                f"hf_hub:nvidia/mit-b0",
            ])
        elif "hrnet" in resolved:
            w_num = name.lower().split("w")[-1] if "w" in name.lower() else "32"
            candidates.extend([
                f"hrnet_w{w_num}",
                f"hrnet_w{w_num}_small",
                "hrnet_w18",
            ])

        last_err = None
        for cand in candidates:
            try:
                return BackboneWithFPN(
                    model_name=cand,
                    pretrained=pretrained,
                    out_channels=out_channels,
                    out_indices=out_indices,
                )
            except Exception as err:
                last_err = err
                continue

        # Attempt fuzzy search within the requested architecture family
        if TIMM_AVAILABLE:
            clean_name = name.lower().replace("-", "_")
            family_prefix = clean_name.split("_")[0]
            search_terms = [clean_name, family_prefix]
            if "segformer" in clean_name or "mit" in clean_name:
                search_terms.extend(["mix_transformer", "segformer", "mit"])
            elif "hrnet" in clean_name:
                search_terms.extend(["hrnet"])
            elif "swin" in clean_name:
                search_terms.extend(["swin"])
            elif "convnext" in clean_name:
                search_terms.extend(["convnext"])

            for term in search_terms:
                matches = timm.list_models(f"*{term}*")
                for match in matches:
                    try:
                        print(f"[BackboneFactory] Trying timm model: '{match}' for requested backbone '{name}'...")
                        return BackboneWithFPN(
                            model_name=match,
                            pretrained=pretrained,
                            out_channels=out_channels,
                            out_indices=out_indices,
                        )
                    except Exception:
                        continue

        # Robust Auto-Fallback to high-performance vision backbones if requested backbone is unavailable
        fallback_models = ["swin_tiny_patch4_window7_224", "swin_tiny", "convnext_tiny", "hrnet_w18", "resnet34"]
        for fb_model in fallback_models:
            try:
                print(f"[BackboneFactory WARNING] Requested backbone '{name}' could not be loaded ({last_err}). Automatically falling back to '{fb_model}'...")
                return BackboneWithFPN(
                    model_name=fb_model,
                    pretrained=pretrained,
                    out_channels=out_channels,
                    out_indices=out_indices,
                )
            except Exception:
                continue

        raise ValueError(f"Could not load backbone '{name}' or any fallback models. Last error: {last_err}")
