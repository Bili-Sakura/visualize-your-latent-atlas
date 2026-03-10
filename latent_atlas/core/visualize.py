"""
t-SNE latent space visualization and uniformity metrics.
"""

import os
from glob import glob
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from safetensors import safe_open


def get_img_to_safefile_map(files: List[str]) -> dict:
    """Create mapping from image index to safetensor file and position."""
    img_to_file = {}
    for safe_file in files:
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            labels = f.get_slice("labels")
            labels_shape = labels.get_shape()
            num_imgs = labels_shape[0]
            cur_len = len(img_to_file)
            for i in range(num_imgs):
                img_to_file[cur_len + i] = {
                    "safe_file": safe_file,
                    "idx_in_file": i,
                }
    return img_to_file


def get_latent_stats(data_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load latent statistics (mean and std) from cache file."""
    latent_stats_cache = os.path.join(data_dir, "latents_stats.pt")
    if not os.path.exists(latent_stats_cache):
        return None, None
    stats = torch.load(latent_stats_cache)
    return stats["mean"], stats["std"]


def get_latent_stats_from_files(safetensor_files: List[str]) -> Optional[Dict[str, torch.Tensor]]:
    """Compute mean/std from safetensor files. Returns dict with 'mean' and 'std'."""
    if not safetensor_files:
        return None
    all_latents = []
    for f in safetensor_files[:10]:  # limit for speed
        with safe_open(f, framework="pt", device="cpu") as sf:
            lat = sf.get_tensor("latents")
            all_latents.append(lat)
    stacked = torch.cat(all_latents, dim=0)
    mean = stacked.mean(dim=(0, 2, 3), keepdim=True)
    std = stacked.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    return {"mean": mean, "std": std}


def sample_latents(
    safetensor_files: List[str],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    sample_num: int = 10000,
) -> torch.Tensor:
    """Sample latent vectors from safetensor files."""
    if latent_mean.dim() >= 2:
        latent_mean = latent_mean.squeeze(dim=(2, 3))
    if latent_std.dim() >= 2:
        latent_std = latent_std.squeeze(dim=(2, 3))

    img_to_file_map = get_img_to_safefile_map(safetensor_files)
    total_imgs = len(img_to_file_map)
    if total_imgs == 0:
        raise ValueError("No images in safetensor files")
    sample_idx = np.random.choice(total_imgs, min(sample_num, total_imgs), replace=False)

    data = []
    for idx in tqdm(sample_idx, desc="Sampling latent vectors"):
        img_info = img_to_file_map[idx]
        safe_file, img_idx = img_info["safe_file"], img_info["idx_in_file"]
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            features = f.get_slice(tensor_key)
            feature = features[img_idx : img_idx + 1]
        h, w = feature.shape[2], feature.shape[3]
        hi = np.random.randint(0, h)
        wi = np.random.randint(0, w)
        pixel_feat = feature[:, :, hi, wi]
        pixel_feat = (pixel_feat - latent_mean) / latent_std
        data.append(pixel_feat)
    return torch.cat(data, dim=0)


def load_latent_data(
    safetensor_files: List[str],
    cache_file: Optional[str] = None,
    sample_num: int = 10000,
) -> torch.Tensor:
    """Load latent vectors from safetensors or cache."""
    if cache_file and os.path.exists(cache_file):
        return torch.load(cache_file)

    data_dir = os.path.dirname(safetensor_files[0]) if safetensor_files else "."
    latent_mean, latent_std = get_latent_stats(data_dir)
    if latent_mean is None or latent_std is None:
        raise ValueError(
            f"latents_stats.pt not found in {data_dir}. "
            "Run extract_features first to generate it."
        )
    latent_mean = latent_mean.squeeze(dim=(2, 3)) if latent_mean.dim() > 2 else latent_mean
    latent_std = latent_std.squeeze(dim=(2, 3)) if latent_std.dim() > 2 else latent_std

    data = sample_latents(safetensor_files, latent_mean, latent_std, sample_num)
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        torch.save(data, cache_file)
    return data


def calculate_uniformity_metrics(tsne_results: np.ndarray) -> Dict[str, float]:
    """Calculate uniformity metrics for t-SNE results."""
    kde = gaussian_kde(tsne_results.T)
    density = kde(tsne_results.T)
    density_mean = np.mean(density)
    density_std = np.std(density)
    density_cv = density_std / (density_mean + 1e-10)
    density_norm = density / (np.sum(density) + 1e-10)
    entropy = -np.sum(density_norm * np.log2(density_norm + 1e-10))
    max_entropy = np.log2(len(density) + 1)
    normalized_entropy = entropy / (max_entropy + 1e-10)
    sorted_density = np.sort(density)
    index = np.arange(1, len(sorted_density) + 1)
    n = len(sorted_density)
    gini = (np.sum((2 * index - n - 1) * sorted_density) / (n * np.sum(sorted_density) + 1e-10))
    return {
        "density_std": float(density_std),
        "density_cv": float(density_cv),
        "normalized_entropy": float(normalized_entropy),
        "gini_coefficient": float(gini),
    }


def plot_tsne_visualization(
    safetensor_files: List[str],
    output_path: str = "tsne_visualization.png",
    n_components: int = 2,
    perplexity: int = 30,
    n_iter: int = 1000,
    cache_file: Optional[str] = None,
    sample_num: int = 10000,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate t-SNE visualization for latent vectors.

    Args:
        safetensor_files: List of safetensor file paths
        output_path: Output image path
        n_components: t-SNE dimensions
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        cache_file: Optional cache for sampled latents
        sample_num: Number of latent vectors to sample

    Returns:
        (tsne_results, metrics)
    """
    data = load_latent_data(safetensor_files, cache_file, sample_num)
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(data.numpy())
    metrics = calculate_uniformity_metrics(tsne_results)

    plt.figure(figsize=(12, 10))
    kde = gaussian_kde(tsne_results.T)
    density = kde(tsne_results.T)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=density, cmap="viridis", alpha=0.6)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return tsne_results, metrics
