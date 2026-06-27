# Text2ImageAttnVisualization

Visualize where a text-to-image diffusion model "looks" while it draws.

This project loads a trained **Stable Diffusion** model (default **SD 1.5**, with
**SDXL base 1.0** also selectable in the web UI) — decomposed into its three
components, the **text encoder** (CLIP), the **VAE**, and the **UNet** — and
captures the attention the UNet computes while denoising, so you can see two
kinds of dependency:

1. **Image → text, Cross-attention** — fix an **image region**
   and see how strongly it attends to each **token** ("what words was this patch
   looking at?"). In the web app the token chips get colored by that weight.

2. **Text → image** — how strongly each **text token** influences on different regions of the generated image ("which pixels rendered the word *cat*?").

3. **Image → image, Self-attention** — how strongly each region of the image
   depends on **other regions** of the image ("when the model drew this patch,
   what else did it look at?").

Most logic lives in plain `.py` modules under `attnviz/`. An interactive Jupyter
notebook (`notebooks/interactive_visualization.ipynb`) lets you click tokens and
image regions and watch the heatmaps update.

## How it works

Modern `diffusers` uses a fused attention kernel that never exposes the
attention matrix. `attnviz` swaps in a custom attention processor
(`CaptureAttnProcessor`) that reimplements the explicit
`softmax(Q·Kᵀ / √d)·V` path and records the softmax weights on the way through.
The numerical output is identical to the stock model; it just keeps the maps.

Maps are averaged over attention heads and over denoising steps, then averaged
across the UNet layers at a chosen latent resolution (16×16 is the most
semantically meaningful for cross-attention).

## Project layout

```
Text2ImageAttnVisualization/
├── README.md
├── requirements.txt
├── main.py                       # thin CLI entry point
├── run_web.py                    # thin web-app entry point
├── notebooks/
│   └── interactive_visualization.ipynb
├── attnviz/                       # core library — one class per file
│   ├── config.py                  # Config (run settings)
│   ├── device_resolver.py         # DeviceResolver (auto cuda/mps/cpu)
│   ├── pipeline_loader.py         # PipelineLoader (text encoder, vae, unet)
│   ├── capture_attn_processor.py  # CaptureAttnProcessor (records softmax)
│   ├── attention_store.py         # AttentionStore (accumulates maps)
│   ├── attention_controller.py    # AttentionController (installs processors)
│   ├── token_decoder.py           # TokenDecoder (prompt -> token labels)
│   ├── generator.py               # Generator (+ GenerationResult)
│   ├── heatmap_renderer.py        # HeatmapRenderer (normalize/upsample/blend)
│   ├── cross_attention_visualizer.py  # token -> image regions
│   ├── self_attention_visualizer.py   # image region -> image regions
│   ├── image_to_text_visualizer.py    # image region -> token weights
│   └── figure_saver.py            # FigureSaver (static PNGs for the CLI)
└── webapp/                        # Flask web UI
    ├── image_codec.py             # ImageCodec (PIL/np -> base64 data URL)
    ├── session_store.py           # SessionStore (LRU of generations)
    ├── model_service.py           # ModelService (lazy load + generate)
    ├── viz_server.py              # VizServer (Flask routes)
    ├── templates/index.html       # page markup
    └── static/{style.css, app.js} # palette styling + interactions
```

## Requirements

- Python 3.9+
- A GPU is recommended but not required — the code auto-detects CUDA → Apple
  Silicon (MPS) → CPU. On CPU, lower `--steps` and `--size`.
- ~5 GB disk for SD 1.5 weights, ~7 GB for SDXL base (downloaded once from
  HuggingFace).
- **SDXL note:** SDXL base 1.0 is much larger than SD 1.5 and the explicit
  attention capture is memory-hungry, so it wants a strong CUDA GPU (≈12–16 GB+
  VRAM at 1024). If you hit out-of-memory, lower the size (e.g. 768) or use
  SD 1.5.

## Install

```bash
cd Text2ImageAttnVisualization
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

The first run downloads the SD 1.5 weights. If the default repo is unavailable,
pass another mirror, e.g. `--model sd-legacy/stable-diffusion-v1-5`.

## Credentials (optional)

Gated models (SDXL, SD 2.1) need a HuggingFace token; ungated SD 1.5 does not.
Copy `.env.example` to `.env` and paste your token:

```bash
cp .env.example .env
# then edit .env:  HF_TOKEN=hf_xxxxxxxx   (get one at huggingface.co/settings/tokens)
```

`.env` is gitignored, so your token is never committed. If the file is missing
or `HF_TOKEN` is blank, the app still runs — it just can't download gated models.
A token set in the shell environment (`export HF_TOKEN=...`) takes precedence.

## Run the CLI

```bash
python main.py --prompt "a cat sitting on a red sofa" --out runs/cat
```

This writes three files into the output folder:

- `generated.png` — the generated image
- `cross_attention.png` — a grid: input + one heatmap per word token
- `self_attention.png` — a grid: input + region-to-region maps at sampled points

Useful flags: `--steps`, `--size`, `--guidance`, `--seed`, `--device`, `--model`.

## Run the web app

```bash
python run_web.py            # then open http://127.0.0.1:8000
```


The page lets you:

0. pick a **model** — Stable Diffusion 1.5 (512) or SDXL base 1.0 (1024); the
   size box jumps to the model's native resolution, and the chosen model loads
   on the next generate (SDXL is a large download and needs a strong GPU);
1. type a **prompt** and generate an image (the model loads on first generate);
2. in **Text → Image** mode, click any **token chip** to heatmap the regions
   that attend to that word (cross-attention);
3. in **Image → Text** mode, **click anywhere on the image** to color each token
   chip by how strongly that region attends to it (image-to-text);
4. in **Image → Image** mode, **click anywhere on the image** to heatmap the
   regions that point attends to (self-attention);
5. drag the **overlay-strength** slider to fade the heatmap in and out
   (cross/self modes);
6. in every mode, use the **UNet layer** dropdown to view a single attention
   layer instead of the average over layers — coarse layers (e.g. `mid.a0 · 8²`)
   track layout, fine layers (`up.3.a2 · 64²`) track edges and detail. Text →
   Image and Image → Text list the cross-attention layers; Image → Image lists
   the self-attention layers;
7. in Image → Text, the start/end marker tokens are **excluded from the score by
   default** (they are attention sinks that flatten the word weights). Tick
   "Include [start]/[end] tokens in score" to fold them back in;
8. in Text → Image, the **Values** dropdown chooses pre-softmax scores
   (`Q·Kᵀ/√d`, the default) or post-softmax probabilities. The softmax is taken
   per query patch, so pre-softmax logits compare more fairly across patches for
   a fixed token; image-to-text keeps post-softmax, where the softmax-over-tokens
   is exactly the right normalization.

Flags: `--host`, `--port`, `--device`, `--model`, `--debug`.

## Run the interactive notebook

```bash
pip install jupyter ipywidgets
jupyter notebook notebooks/interactive_visualization.ipynb
```

## Using it as a library

```python
from attnviz import Config, DeviceResolver, PipelineLoader, Generator
from attnviz import CrossAttentionVisualizer, SelfAttentionVisualizer

config = Config(num_inference_steps=30, seed=0)
resolver = DeviceResolver(config.device, config.dtype)
loader = PipelineLoader(config, resolver); loader.load()

result = Generator(config, loader).generate("a cat on a red sofa")

cross = CrossAttentionVisualizer(result)
heat = cross.token_heatmap(cross.word_token_indices()[0])   # numpy [H, W]

self_attn = SelfAttentionVisualizer(result)
region = self_attn.region_heatmap(x=256, y=256)             # numpy [H, W]
```

## Notes & limitations

- Self-attention maps grow with resolution⁴, so by default only latent grids
  up to 32×32 are kept (`Config.self_attn_max_res`). Raise it for finer maps at
  the cost of memory.
- With classifier-free guidance the UNet runs an unconditional and a
  conditional pass; the visualizers use the conditional one.
- The capture path is the classic (non-flash) attention, so it is a bit slower
  and uses more memory than a normal inference run.
