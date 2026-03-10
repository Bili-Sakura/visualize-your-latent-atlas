#!/usr/bin/env python3
"""
CLI for t-SNE latent visualization.
Usage:
  python scripts/visualize.py --latent-dir outputs/latents --output tsne.png
"""

import argparse
from glob import glob
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_atlas.core.visualize import plot_tsne_visualization


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of VAE latents")
    parser.add_argument("--latent-dir", "-d", required=True, help="Directory with *.safetensors")
    parser.add_argument("--output", "-o", default="tsne_visualization.png")
    parser.add_argument("--cache", default=None, help="Cache file for sampled latents")
    parser.add_argument("--sample-num", type=int, default=10000)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--n-iter", type=int, default=1000)
    args = parser.parse_args()

    files = sorted(glob(f"{args.latent_dir}/*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {args.latent_dir}")

    tsne_results, metrics = plot_tsne_visualization(
        safetensor_files=files,
        output_path=args.output,
        cache_file=args.cache,
        sample_num=args.sample_num,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
    )
    print(f"Saved to {args.output}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
