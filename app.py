"""
Hugging Face Spaces entry point.
Run: python app.py
Or on HF: the Space uses this as the main Gradio app.
"""

import os
import sys

# Ensure we find config in project root when running from app.py
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from latent_atlas.web.gradio_app import create_app, launch_app

# For Hugging Face Spaces: create_app(hf_spaces=True)
# For local: create_app(hf_spaces=False)
IS_SPACES = os.environ.get("SPACE_ID") is not None or os.environ.get("HF_HOME")

app = create_app(
    title="Visualize Your Latent Atlas",
    desc="Extract VAE latents from images and visualize latent space with t-SNE.",
    hf_spaces=IS_SPACES,
)

if __name__ == "__main__":
    app.launch(
        share=not IS_SPACES,
        server_port=int(os.environ.get("PORT", 7860)),
    )
