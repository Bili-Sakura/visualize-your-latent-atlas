"""
VAE model loading utilities.
Unified interface for diffusers-based VAEs. Supports HuggingFace Hub and local paths.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from latent_atlas.config import get_config, VAEConfig, PROJECT_ROOT

# Diffusers VAE classes
try:
    from diffusers import (
        AutoencoderKL,
        AutoencoderDC,
        AutoencoderKLQwenImage,
        AutoencoderKLFlux2,
    )
    VAE_CLASSES = {
        "AutoencoderKL": AutoencoderKL,
        "AutoencoderDC": AutoencoderDC,
        "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
        "AutoencoderKLFlux2": AutoencoderKLFlux2,
    }
except ImportError:
    from diffusers import AutoencoderKL
    VAE_CLASSES = {"AutoencoderKL": AutoencoderKL}


class VAEWrapper(nn.Module):
    """Unified wrapper for VAE architectures with consistent encode/decode."""

    def __init__(
        self,
        model: nn.Module,
        config: VAEConfig,
        vae_class_name: str = "AutoencoderKL",
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.scaling_factor = config.scaling_factor
        self.vae_class_name = vae_class_name
        self._original_shape = None

    def _get_pad_factor(self) -> int:
        return 32 if "SANA" in self.config.name else 2

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_factor = self._get_pad_factor()
        pad_h = (pad_factor - h % pad_factor) % pad_factor
        pad_w = (pad_factor - w % pad_factor) % pad_factor
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent. Input: (B, 3, H, W) in [-1, 1]."""
        self._original_shape = x.shape
        x = self._pad_input(x)
        if self.vae_class_name == "AutoencoderKLQwenImage" and x.dim() == 4:
            x = x.unsqueeze(2)
        encoded = self.model.encode(x)
        z = (
            encoded.latent_dist.sample()
            if hasattr(encoded, "latent_dist")
            else (
                encoded.latent
                if hasattr(encoded, "latent")
                else encoded
            )
        )
        if self.vae_class_name == "AutoencoderKLQwenImage" and z.dim() == 5:
            z = z.squeeze(2)
        return z * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor, original_shape=None) -> torch.Tensor:
        """Decode latents to images. Output: (B, 3, H, W) in [-1, 1]."""
        z = z / self.scaling_factor
        if self.vae_class_name == "AutoencoderKLQwenImage" and z.dim() == 4:
            z = z.unsqueeze(2)
        decoded = self.model.decode(z)
        result = decoded.sample if hasattr(decoded, "sample") else decoded
        if self.vae_class_name == "AutoencoderKLQwenImage" and result.dim() == 5:
            result = result.squeeze(2)
        target_shape = original_shape or self._original_shape
        if target_shape is not None and result.shape[-2:] != target_shape[-2:]:
            result = result[..., : target_shape[-2], : target_shape[-1]]
        return result

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode (full reconstruction)."""
        return self.decode(self.encode(x), original_shape=x.shape)


def _is_hub_id(path: str) -> bool:
    """Check if path is a HuggingFace Hub ID (org/model)."""
    return "/" in path and not path.startswith("/") and not path.startswith(".")

def _resolve_path(path: str):
    """
    Resolve path: Hub IDs pass through; local paths resolved.
    Returns str for Hub IDs, Path for local.
    """
    if _is_hub_id(path):
        return path  # diffusers from_pretrained handles Hub IDs
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    resolved = PROJECT_ROOT / path
    return str(resolved) if resolved.exists() else str(p)


def _get_dtype(device: str, dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_vae(
    model_name: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
) -> VAEWrapper:
    """
    Load a VAE model.

    Args:
        model_name: Registered name from config.yaml (e.g., "SD21-VAE")
        pretrained_path: Override path/Hub ID (bypasses registry)
        device: Target device
        dtype: Model dtype (auto if None)

    Returns:
        VAEWrapper instance
    """
    if pretrained_path is not None:
        vae_config = VAEConfig(
            name="custom",
            pretrained_path=pretrained_path,
        )
        load_path = _resolve_path(pretrained_path)
        ckpt_path = Path(load_path) if not _is_hub_id(pretrained_path) else None
    else:
        if model_name is None:
            raise ValueError("Provide model_name or pretrained_path")
        cfg = get_config()
        if model_name not in cfg["vaes"]:
            raise ValueError(
                f"Unknown VAE: {model_name}. Available: {list(cfg['vaes'].keys())}"
            )
        vae_config = cfg["vaes"][model_name]
        load_path = _resolve_path(vae_config.pretrained_path)
        ckpt_path = Path(load_path) if not _is_hub_id(vae_config.pretrained_path) else None

    vae_class = VAE_CLASSES.get("AutoencoderKL")
    vae_class_name = "AutoencoderKL"

    config_json = None
    if ckpt_path is not None:
        config_json = ckpt_path / "config.json"
        if vae_config.subfolder:
            config_json = ckpt_path / vae_config.subfolder / "config.json"

    if config_json is not None and config_json.exists():
        with open(config_json, "r") as f:
            model_cfg = json.load(f)
        class_name = model_cfg.get("_class_name", "AutoencoderKL")
        if class_name in VAE_CLASSES:
            vae_class = VAE_CLASSES[class_name]
            vae_class_name = class_name
        vae_config = VAEConfig(
            name=vae_config.name,
            pretrained_path=vae_config.pretrained_path,
            scaling_factor=model_cfg.get("scaling_factor", vae_config.scaling_factor),
            latent_channels=model_cfg.get("latent_channels", vae_config.latent_channels),
            spatial_compression=vae_config.spatial_compression,
            subfolder=vae_config.subfolder,
        )

    dtype = _get_dtype(device, dtype)
    load_kwargs = dict(
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )

    load_path = _resolve_path(vae_config.pretrained_path) if pretrained_path is None else _resolve_path(pretrained_path)
    if vae_config.subfolder:
        model = vae_class.from_pretrained(
            load_path, subfolder=vae_config.subfolder, **load_kwargs
        )
    else:
        model = vae_class.from_pretrained(load_path, **load_kwargs)

    model = model.to(device).eval()
    return VAEWrapper(model, vae_config, vae_class_name)
