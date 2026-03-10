"""
Dataset loaders with flexible registration support.
Supports ImageFolder structure and custom paths.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from latent_atlas.config import get_config, PROJECT_ROOT, DatasetConfig


def get_image_transform(
    image_size: int,
    normalize: bool = True,
    p_hflip: float = 0.0,
) -> transforms.Compose:
    """
    Get standard image transform for VAE encoding.
    Output: (B, 3, H, W) in [-1, 1].
    """
    t = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalize:
        t.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    if p_hflip > 0:
        t.append(transforms.RandomHorizontalFlip(p=p_hflip))
    return transforms.Compose(t)


class ImageFolderDataset(Dataset):
    """
    ImageFolder-style dataset with optional size and transform.
    root/class_name/img.jpg
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        split_file: Optional[str] = None,
    ):
        self.root = self._resolve_path(root)
        self.image_size = image_size
        self.transform = transform or get_image_transform(image_size)

        split_filenames = self._load_split_file(split_file) if split_file else None
        self.samples, self.classes = self._scan_directory(split_filenames)

        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path) and os.path.exists(path):
            return path
        resolved = PROJECT_ROOT / path
        return str(resolved) if resolved.exists() else path

    def _load_split_file(self, split_file: str) -> Optional[set]:
        path = self._resolve_path(split_file)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())

    def _scan_directory(self, split_filenames: Optional[set]) -> Tuple[List[Tuple[str, int]], List[str]]:
        if not os.path.exists(self.root):
            raise ValueError(f"Dataset root not found: {self.root}")

        all_classes = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )
        if not all_classes:
            # Flat folder (no class subdirs)
            valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            samples = []
            for f in os.listdir(self.root):
                if f.lower().endswith(valid_exts):
                    if split_filenames and f not in split_filenames:
                        continue
                    samples.append((os.path.join(self.root, f), 0))
            return samples, ["default"]

        samples = []
        valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        for idx, class_name in enumerate(all_classes):
            class_dir = os.path.join(self.root, class_name)
            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(valid_exts):
                    continue
                if split_filenames and img_name not in split_filenames:
                    continue
                samples.append((os.path.join(class_dir, img_name), idx))
        return samples, all_classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_dataset(
    dataset_name: Optional[str] = None,
    root: Optional[str] = None,
    image_size: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    split_file: Optional[str] = None,
) -> Tuple[Dataset, DataLoader]:
    """
    Load dataset by registry name or direct path.

    Args:
        dataset_name: Registered name from config.yaml
        root: Override root path (bypasses registry)
        image_size: Override image size
        batch_size: DataLoader batch size
        num_workers: DataLoader workers
        split_file: Optional split file

    Returns:
        (dataset, dataloader)
    """
    if root is not None:
        ds_config = DatasetConfig(
            name="custom",
            root=root,
            image_size=image_size or 256,
        )
        dataset = ImageFolderDataset(
            root=ds_config.root,
            image_size=ds_config.image_size,
            split_file=split_file,
        )
    else:
        if dataset_name is None:
            raise ValueError("Provide dataset_name or root")
        cfg = get_config()
        if dataset_name not in cfg["datasets"]:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(cfg['datasets'].keys())}"
            )
        ds_config = cfg["datasets"][dataset_name]
        dataset = ImageFolderDataset(
            root=ds_config.root,
            image_size=image_size or ds_config.image_size,
            split_file=split_file or ds_config.split_file,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader
