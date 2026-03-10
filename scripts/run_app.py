#!/usr/bin/env python3
"""Launch Gradio app locally."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_atlas.web.gradio_app import create_app, launch_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--hf-style", action="store_true", help="HF Spaces style (no share)")
    args = parser.parse_args()
    launch_app(hf_spaces=args.hf_style, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
