# Registry Format Specification

Visualize Your Latent Atlas uses a **config-driven registry** for VAE models and datasets. No code changes required—add entries to `config.yaml`.

## VAE Registry

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pretrained_path` | str | Yes | HuggingFace Hub ID (e.g. `stabilityai/sd-vae-ft-mse`) or local path |
| `scaling_factor` | float | No | Latent scaling (default: 0.18215) |
| `latent_channels` | int | No | Latent channels (default: 4) |
| `spatial_compression` | int | No | Downsample factor (default: 8) |
| `subfolder` | str | No | Subfolder inside checkpoint (e.g. `vae`) |

### Architecture Selection

The loader reads `config.json` in the checkpoint. The `_class_name` field selects the diffusers class:

- `AutoencoderKL` (default)
- `AutoencoderDC`
- `AutoencoderKLQwenImage`
- `AutoencoderKLFlux2`

### Example

```yaml
vaes:
  SD21-VAE:
    pretrained_path: stabilityai/sd-vae-ft-mse
    scaling_factor: 0.18215
    latent_channels: 4
    spatial_compression: 8

  FLUX1-VAE:
    pretrained_path: black-forest-labs/FLUX.1-schnell
    subfolder: vae
    scaling_factor: 0.3611
    latent_channels: 16
    spatial_compression: 8

  Local-VAE:
    pretrained_path: ./models/my_vae
    scaling_factor: 0.18215
```

## Dataset Registry

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `root` | str | Yes | Path to ImageFolder root (`root/class_name/img.jpg`) or flat image dir |
| `image_size` | int | No | Resize/center crop size (default: 256) |
| `num_classes` | int | No | Auto if null |
| `split_file` | str | No | One filename per line to restrict samples |

### Layouts

1. **ImageFolder**: `root/class_a/img1.jpg`, `root/class_b/img2.jpg`
2. **Flat**: `root/img1.jpg`, `root/img2.jpg` (single class "default")

### Example

```yaml
datasets:
  imagenet_sample:
    root: ./data/imagenet
    image_size: 256
    num_classes: null

  custom:
    root: /absolute/path/to/images
    image_size: 512
    split_file: ./splits/train.txt
```

## Defaults

```yaml
defaults:
  sample_num: 10000
  batch_size: 16
  num_workers: 4
  seed: 42
  tsne_perplexity: 30
  tsne_iterations: 1000
```

## Adding Custom Entries

1. Open `config.yaml`
2. Add a new key under `vaes:` or `datasets:`
3. Fill in the fields
4. Restart the app or reload config

No Python code changes needed.
