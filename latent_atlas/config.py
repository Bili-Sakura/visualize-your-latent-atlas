"""
Configuration loader for Visualize Your Latent Atlas.
Loads VAE and dataset registry from config.yaml.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class VAEConfig:
    """VAE model configuration."""
    name: str
    pretrained_path: str  # local path or HuggingFace Hub ID
    scaling_factor: float = 0.18215
    latent_channels: int = 4
    spatial_compression: int = 8
    subfolder: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    root: str
    image_size: int = 256
    num_classes: Optional[int] = None
    split_file: Optional[str] = None


@dataclass
class DefaultsConfig:
    """Default settings for extraction and visualization."""
    sample_num: int = 10000
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    tsne_perplexity: int = 30
    tsne_iterations: int = 1000


def load_yaml_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        fallback = PROJECT_ROOT / "config.example.yaml"
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Copy config.example.yaml to config.yaml and customize."
            )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and parse configuration. Returns a dict with:
    - vaes: Dict[str, VAEConfig]
    - datasets: Dict[str, DatasetConfig]
    - defaults: DefaultsConfig
    """
    path = config_path or DEFAULT_CONFIG_PATH
    raw = load_yaml_config(path)

    vaes: Dict[str, VAEConfig] = {}
    for name, cfg in raw.get("vaes", {}).items():
        vaes[name] = VAEConfig(
            name=name,
            pretrained_path=cfg.get("pretrained_path", ""),
            scaling_factor=float(cfg.get("scaling_factor", 0.18215)),
            latent_channels=int(cfg.get("latent_channels", 4)),
            spatial_compression=int(cfg.get("spatial_compression", 8)),
            subfolder=cfg.get("subfolder"),
        )

    datasets: Dict[str, DatasetConfig] = {}
    for name, cfg in raw.get("datasets", {}).items():
        datasets[name] = DatasetConfig(
            name=name,
            root=cfg.get("root", ""),
            image_size=int(cfg.get("image_size", 256)),
            num_classes=cfg.get("num_classes"),
            split_file=cfg.get("split_file"),
        )

    def_raw = raw.get("defaults", {})
    defaults = DefaultsConfig(
        sample_num=int(def_raw.get("sample_num", 10000)),
        batch_size=int(def_raw.get("batch_size", 16)),
        num_workers=int(def_raw.get("num_workers", 4)),
        seed=int(def_raw.get("seed", 42)),
        tsne_perplexity=int(def_raw.get("tsne_perplexity", 30)),
        tsne_iterations=int(def_raw.get("tsne_iterations", 1000)),
    )

    return {
        "vaes": vaes,
        "datasets": datasets,
        "defaults": defaults,
    }


_config: Optional[Dict[str, Any]] = None


def get_config(config_path: Optional[Path] = None, reload: bool = False) -> Dict[str, Any]:
    """Get the global configuration (lazy-loaded)."""
    global _config
    if _config is None or reload:
        _config = load_config(config_path)
    return _config
