# Deploy on Hugging Face Spaces

1. Create a new Space with **Gradio** SDK.
2. Add `app.py` as the application file (Space will run `gradio app.py`).
3. Add `requirements.txt` in the root.
4. The app auto-detects Spaces via `SPACE_ID` / `HF_HOME` and disables `share` for public deployment.

## Space YAML (optional)

```yaml
sdk: gradio
app_file: app.py
```

## Hardware

- **CPU**: Visualization tab works (uses pre-computed latents).
- **GPU (T4 or better)**: Extract tab works for encoding images with VAE.

## Demo data

For CPU-only Spaces, you can use pre-extracted latents from:
- [hustvl/va-vae-imagenet256-experimental-variants](https://huggingface.co/hustvl/va-vae-imagenet256-experimental-variants)

Download a latent directory and place `*.safetensors` and `latents_stats.pt` in a folder, then use the "3. From pre-computed latents" tab.
