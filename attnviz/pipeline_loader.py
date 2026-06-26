"""Load a Stable Diffusion pipeline and expose its three components."""

from __future__ import annotations

from .config import Config
from .device_resolver import DeviceResolver


class PipelineLoader:
    """Builds an SD pipeline and surfaces text encoder, VAE and UNet.

    A text-to-image diffusion model is three trained networks working
    together:

    * the *text encoder* (CLIP) turns the prompt into token embeddings,
    * the *UNet* denoises a latent image, attending to those embeddings,
    * the *VAE* decodes the final latent into pixels.

    Two families are supported through ``Config.architecture``:

    * ``"sd"`` — Stable Diffusion 1.5 / 2.1 (single CLIP text encoder),
    * ``"sdxl"`` — Stable Diffusion XL base 1.0 (two text encoders, larger
      UNet). SDXL exposes the same ``.text_encoder`` / ``.tokenizer`` (CLIP
      ViT-L) used for token labels, plus the same ``attn1`` / ``attn2``
      cross-attention layers, so the capture + visualization stack is
      unchanged.
    """

    def __init__(self, config: Config, resolver: DeviceResolver):
        self._config = config
        self._resolver = resolver
        self._pipe = None

    def load(self):
        """Download/load weights and move them onto the chosen device."""
        pipe = self._build_pipe()
        pipe = pipe.to(self._resolver.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _build_pipe(self):
        if self._config.architecture == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            return StableDiffusionXLPipeline.from_pretrained(
                self._config.model_id,
                torch_dtype=self._resolver.dtype,
                add_watermarker=False,
                use_safetensors=True,
            )
        from diffusers import StableDiffusionPipeline
        return StableDiffusionPipeline.from_pretrained(
            self._config.model_id,
            torch_dtype=self._resolver.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    @property
    def pipe(self):
        self._require_loaded()
        return self._pipe

    @property
    def text_encoder(self):
        self._require_loaded()
        return self._pipe.text_encoder

    @property
    def tokenizer(self):
        self._require_loaded()
        return self._pipe.tokenizer

    @property
    def vae(self):
        self._require_loaded()
        return self._pipe.vae

    @property
    def unet(self):
        self._require_loaded()
        return self._pipe.unet

    def _require_loaded(self) -> None:
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded yet — call load() first.")
