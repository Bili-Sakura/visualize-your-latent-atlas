#!/usr/bin/env python3
"""
CLI for feature extraction.
Usage:
  python scripts/extract_features.py --vae SD21-VAE --data /path/to/images --output outputs/latents
"""

import argparse
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_atlas.core.extract import extract_features


def main():
    parser = argparse.ArgumentParser(description="Extract VAE latents from images")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--vae", default="SD21-VAE", help="VAE name from config.yaml")
    parser.add_argument("--vae-path", default=None, help="Override: path or HuggingFace Hub ID")
    parser.add_argument("--data", "-d", required=True, help="Image path (ImageFolder root)")
    parser.add_argument("--dataset", default=None, help="Dataset name from config.yaml")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=None, help="Limit samples (default: all)")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_features(
        output_dir=args.output,
        vae_name=args.vae if not args.vae_path else None,
        vae_path=args.vae_path,
        dataset_name=args.dataset,
        data_root=args.data,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        shard_size=args.shard_size,
        device=args.device,
        seed=args.seed,
    )
    print(f"Saved latents to {args.output}")


if __name__ == "__main__":
    main()
