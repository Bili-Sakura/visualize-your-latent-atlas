#!/usr/bin/env python3
"""
Run t-SNE visualization on latent_demos caches from LightningDiT.
https://github.com/hustvl/LightningDiT
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from latent_atlas.core.visualize import plot_tsne_visualization

DEMOS = ROOT / "latent_demos"

def main():
    demos = [
        ("latents_cache_f16d32.pt", "latent_tsne_f16d32.png"),
        ("latents_cache_f16d32_vfdinov2.pt", "latent_tsne_f16d32_vfdinov2.png"),
    ]
    for cache_name, out_name in demos:
        cache = DEMOS / cache_name
        if not cache.exists():
            print(f"Skip (missing): {cache}")
            continue
        out = DEMOS / out_name
        print(f"Visualizing {cache_name} -> {out_name}")
        tsne_results, metrics = plot_tsne_visualization(
            safetensor_files=[],
            output_path=str(out),
            cache_file=str(cache),
            sample_num=10000,
            perplexity=30,
            n_iter=1000,
        )
        print(f"  Saved {out}")
        print(f"  Metrics: {metrics}")

if __name__ == "__main__":
    main()
