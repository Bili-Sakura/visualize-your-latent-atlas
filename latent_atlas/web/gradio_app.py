"""
Gradio Web UI for Visualize Your Latent Atlas.
Supports both local and Hugging Face Spaces deployment.
"""

import os
import tempfile
from glob import glob
from pathlib import Path

import gradio as gr

# Optional imports for full functionality
try:
    from latent_atlas.registry import list_vaes, list_datasets, get_defaults
    from latent_atlas.models import load_vae
    from latent_atlas.core.extract import extract_features
    from latent_atlas.core.visualize import plot_tsne_visualization, load_latent_data
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False


def _get_vae_choices():
    if not HAS_BACKEND:
        return ["SD21-VAE (demo)"]
    try:
        choices = list_vaes()
        return choices if choices else ["SD21-VAE (demo)"]
    except Exception:
        return ["SD21-VAE (demo)"]


def _get_dataset_choices():
    if not HAS_BACKEND:
        return ["Custom path"]
    try:
        choices = list_datasets()
        return choices if choices else ["Custom path"]
    except Exception:
        return ["Custom path"]


def run_extraction(
    vae_name: str,
    data_path: str,
    output_path: str,
    image_size: int,
    batch_size: int,
    sample_limit: int,
    progress=gr.Progress,
):
    """Run feature extraction from images."""
    if not HAS_BACKEND:
        return "Backend not available. Install: pip install latent-atlas[full]"
    if not data_path or not os.path.isdir(data_path):
        return f"Invalid data path: {data_path}"
    out_dir = output_path or tempfile.mkdtemp(prefix="latent_atlas_")
    os.makedirs(out_dir, exist_ok=True)
    try:
        extract_features(
            output_dir=out_dir,
            vae_name=vae_name if vae_name in _get_vae_choices() else None,
            vae_path=None,
            data_root=data_path,
            image_size=image_size,
            batch_size=batch_size,
            num_samples=sample_limit if sample_limit and sample_limit > 0 else None,
        )
        files = glob(os.path.join(out_dir, "*.safetensors"))
        return f"Extracted {len(files)} shard(s) to {out_dir}"
    except Exception as e:
        return f"Error: {e}"


def run_visualization(
    latent_dir: str,
    cache_file: str,
    output_path: str,
    sample_num: int,
    perplexity: int,
    n_iter: int,
    progress=gr.Progress,
):
    """Run t-SNE visualization on extracted latents."""
    if not HAS_BACKEND:
        return None, "Backend not available."
    if not latent_dir or not os.path.isdir(latent_dir):
        return None, f"Invalid latent directory: {latent_dir}"
    files = sorted(glob(os.path.join(latent_dir, "*.safetensors")))
    if not files:
        return None, f"No .safetensors found in {latent_dir}"
    out_path = output_path or os.path.join(latent_dir, "tsne_visualization.png")
    cache = cache_file if cache_file and cache_file.strip() else None
    try:
        tsne_results, metrics = plot_tsne_visualization(
            safetensor_files=files,
            output_path=out_path,
            perplexity=perplexity,
            n_iter=n_iter,
            cache_file=cache,
            sample_num=sample_num,
        )
        msg = f"Saved to {out_path}\nMetrics: {metrics}"
        return out_path, msg
    except Exception as e:
        return None, f"Error: {e}"


def create_app(
    title: str = "Visualize Your Latent Atlas",
    desc: str = "Extract VAE latents and visualize latent space with t-SNE.",
    hf_spaces: bool = False,
):
    """
    Create Gradio app. Set hf_spaces=True for Hugging Face Spaces (share=False, etc.).
    """
    vae_choices = _get_vae_choices()
    dataset_choices = _get_dataset_choices()
    try:
        defaults = get_defaults() if HAS_BACKEND else None
    except Exception:
        defaults = None
    sample_num = defaults.sample_num if defaults else 10000
    batch_size = defaults.batch_size if defaults else 16
    perplexity = defaults.tsne_perplexity if defaults else 30
    n_iter = defaults.tsne_iterations if defaults else 1000

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {title}\n{desc}")

        with gr.Tabs():
            # Tab 1: Extract features
            with gr.Tab("1. Extract Latents"):
                gr.Markdown("Encode images with a VAE and save latents to safetensors.")
                with gr.Row():
                    ext_vae = gr.Dropdown(
                        choices=vae_choices,
                        value=vae_choices[0] if vae_choices else None,
                        label="VAE Model",
                    )
                    ext_data = gr.Textbox(
                        label="Data path (ImageFolder root)",
                        placeholder="/path/to/images or ./data",
                    )
                with gr.Row():
                    ext_out = gr.Textbox(label="Output directory", placeholder="outputs/latents")
                    ext_size = gr.Number(value=256, label="Image size", precision=0)
                with gr.Row():
                    ext_batch = gr.Number(value=batch_size, label="Batch size", precision=0)
                    ext_limit = gr.Number(value=0, label="Sample limit (0=all)", precision=0)
                ext_btn = gr.Button("Extract", variant="primary")
                ext_status = gr.Textbox(label="Status", interactive=False)
                ext_btn.click(
                    fn=run_extraction,
                    inputs=[ext_vae, ext_data, ext_out, ext_size, ext_batch, ext_limit],
                    outputs=[ext_status],
                )

            # Tab 2: Visualize
            with gr.Tab("2. Visualize Latent Space"):
                gr.Markdown("Run t-SNE on extracted latents and generate a scatter plot.")
                with gr.Row():
                    vis_dir = gr.Textbox(
                        label="Latent directory (contains *.safetensors)",
                        placeholder="outputs/latents",
                    )
                    vis_cache = gr.Textbox(
                        label="Cache file (optional)",
                        placeholder="cache.pt",
                    )
                with gr.Row():
                    vis_out = gr.Textbox(label="Output image path", placeholder="tsne.png")
                    vis_sample = gr.Number(value=sample_num, label="Samples", precision=0)
                with gr.Row():
                    vis_perplexity = gr.Number(value=perplexity, label="t-SNE perplexity", precision=0)
                    vis_iter = gr.Number(value=n_iter, label="t-SNE iterations", precision=0)
                vis_btn = gr.Button("Visualize", variant="primary")
                vis_image = gr.Image(label="t-SNE plot")
                vis_status = gr.Textbox(label="Status", interactive=False)
                vis_btn.click(
                    fn=run_visualization,
                    inputs=[
                        vis_dir,
                        vis_cache,
                        vis_out,
                        vis_sample,
                        vis_perplexity,
                        vis_iter,
                    ],
                    outputs=[vis_image, vis_status],
                )

            # Tab 3: Quick demo (pre-computed)
            with gr.Tab("3. From pre-computed latents"):
                gr.Markdown(
                    "Use pre-extracted latents (e.g. from HuggingFace). "
                    "Provide a directory with `*.safetensors` and `latents_stats.pt`."
                )
                demo_dir = gr.Textbox(
                    label="Latent directory",
                    placeholder="path/to/latents",
                )
                demo_btn = gr.Button("Visualize", variant="primary")
                demo_image = gr.Image(label="t-SNE plot")
                demo_status = gr.Textbox(label="Status", interactive=False)

                def _demo_vis(d):
                    return run_visualization(
                        d, "", "", sample_num, perplexity, n_iter
                    )

                demo_btn.click(
                    fn=_demo_vis,
                    inputs=[demo_dir],
                    outputs=[demo_image, demo_status],
                )

        gr.Markdown("---\n*Config: `config.yaml` | Add VAEs/datasets there.*")

    return app


def launch_app(
    hf_spaces: bool = False,
    share: bool = False,
    server_port: int = 7860,
):
    """Launch the Gradio app."""
    app = create_app(hf_spaces=hf_spaces)
    app.launch(
        share=share and not hf_spaces,
        server_port=server_port,
    )


