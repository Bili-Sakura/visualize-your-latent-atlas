"""
Visualize Your Latent Atlas - Explore VAE latent spaces with t-SNE.

Provides:
- VAE and dataset registry (config-driven)
- Feature extraction from images
- t-SNE latent space visualization
- Gradio web UI (local and Hugging Face Spaces)
"""

from latent_atlas.config import get_config, load_config, VAEConfig, DatasetConfig
from latent_atlas.registry import list_vaes, list_datasets, get_vae_config, get_dataset_config
from latent_atlas.models import load_vae
from latent_atlas.datasets import load_dataset
from latent_atlas.core.visualize import plot_tsne_visualization, calculate_uniformity_metrics
from latent_atlas.core.extract import extract_features

__version__ = "0.1.0"

__all__ = [
    "get_config",
    "load_config",
    "VAEConfig",
    "DatasetConfig",
    "list_vaes",
    "list_datasets",
    "get_vae_config",
    "get_dataset_config",
    "load_vae",
    "load_dataset",
    "plot_tsne_visualization",
    "calculate_uniformity_metrics",
    "extract_features",
]
