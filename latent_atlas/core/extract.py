"""
Feature extraction from images via VAE.
Saves latents to safetensors with optional latent stats.
"""

import os
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from tqdm import tqdm

from latent_atlas.models import load_vae
from latent_atlas.datasets import load_dataset
from latent_atlas.config import get_config


def extract_features(
    output_dir: str,
    vae_name: Optional[str] = None,
    vae_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    data_root: Optional[str] = None,
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    num_samples: Optional[int] = None,
    shard_size: int = 10000,
    device: str = "cuda",
    seed: int = 42,
) -> List[str]:
    """
    Extract VAE latents from images and save to safetensors.

    Args:
        output_dir: Directory to save latents
        vae_name: Registered VAE name
        vae_path: Override VAE path/Hub ID
        dataset_name: Registered dataset name
        data_root: Override data path
        image_size: Image size for encoding
        batch_size: Batch size
        num_workers: DataLoader workers
        num_samples: Max samples (None = all)
        shard_size: Samples per safetensor file
        device: Device
        seed: Random seed

    Returns:
        List of saved safetensor file paths
    """
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    vae = load_vae(
        model_name=vae_name,
        pretrained_path=vae_path,
        device=device,
    )

    _, loader = load_dataset(
        dataset_name=dataset_name,
        root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    saved_files = []
    latents_buf = []
    latents_flip_buf = []
    labels_buf = []
    run_count = 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Extracting")):
        x = x.to(device)
        z = vae.encode(x).detach().cpu()
        z_flip = vae.encode(torch.flip(x, dims=[-1])).detach().cpu()

        latents_buf.append(z)
        latents_flip_buf.append(z_flip)
        labels_buf.append(y)
        run_count += x.shape[0]

        if num_samples is not None and run_count >= num_samples:
            break

        if sum(t.shape[0] for t in latents_buf) >= shard_size:
            latents = torch.cat(latents_buf, dim=0)[:shard_size]
            latents_flip = torch.cat(latents_flip_buf, dim=0)[:shard_size]
            labels = torch.cat(labels_buf, dim=0)[:shard_size]
            save_dict = {
                "latents": latents,
                "latents_flip": latents_flip,
                "labels": labels,
            }
            fname = os.path.join(output_dir, f"latents_shard{len(saved_files):03d}.safetensors")
            save_file(
                save_dict,
                fname,
                metadata={
                    "total_size": str(latents.shape[0]),
                    "dtype": str(latents.dtype),
                },
            )
            saved_files.append(fname)
            excess = sum(t.shape[0] for t in latents_buf) - shard_size
            if excess > 0:
                latents_buf = [latents_buf[-1][-excess:]]
                latents_flip_buf = [latents_flip_buf[-1][-excess:]]
                labels_buf = [labels_buf[-1][-excess:]]
            else:
                latents_buf = []
                latents_flip_buf = []
                labels_buf = []

    if latents_buf:
        latents = torch.cat(latents_buf, dim=0)
        latents_flip = torch.cat(latents_flip_buf, dim=0)
        labels = torch.cat(labels_buf, dim=0)
        save_dict = {
            "latents": latents,
            "latents_flip": latents_flip,
            "labels": labels,
        }
        fname = os.path.join(output_dir, f"latents_shard{len(saved_files):03d}.safetensors")
        save_file(save_dict, fname, metadata={"total_size": str(latents.shape[0])})
        saved_files.append(fname)

    # Compute and save latent stats
    from latent_atlas.core.visualize import get_latent_stats_from_files
    st_files = [f for f in saved_files if f.endswith(".safetensors")]
    stats = get_latent_stats_from_files(st_files)
    if stats is not None:
        torch.save(stats, os.path.join(output_dir, "latents_stats.pt"))

    return saved_files
