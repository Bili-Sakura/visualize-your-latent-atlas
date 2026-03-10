"""
Registry module for VAE models and datasets.
Uses config-driven registration (YAML); no decorators.
"""

from latent_atlas.config import get_config
from latent_atlas.config import VAEConfig, DatasetConfig


def list_vaes() -> list:
    """Return list of registered VAE model names."""
    cfg = get_config()
    return list(cfg["vaes"].keys())


def list_datasets() -> list:
    """Return list of registered dataset names."""
    cfg = get_config()
    return list(cfg["datasets"].keys())


def get_vae_config(name: str) -> VAEConfig:
    """Get VAE config by name."""
    cfg = get_config()
    if name not in cfg["vaes"]:
        raise ValueError(
            f"Unknown VAE: {name}. Available: {list_vaes()}"
        )
    return cfg["vaes"][name]


def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset config by name."""
    cfg = get_config()
    if name not in cfg["datasets"]:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list_datasets()}"
        )
    return cfg["datasets"][name]


def get_defaults():
    """Get default settings."""
    return get_config()["defaults"]


__all__ = [
    "list_vaes",
    "list_datasets",
    "get_vae_config",
    "get_dataset_config",
    "get_defaults",
]
