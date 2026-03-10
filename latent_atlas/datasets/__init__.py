"""Dataset loading with registry support."""

from latent_atlas.datasets.loaders import (
    load_dataset,
    ImageFolderDataset,
    get_image_transform,
)

__all__ = ["load_dataset", "ImageFolderDataset", "get_image_transform"]
