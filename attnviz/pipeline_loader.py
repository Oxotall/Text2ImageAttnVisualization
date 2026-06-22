"""Load the Stable Diffusion 1.5 pipeline and expose its three components."""

from __future__ import annotations

import torch
from diffusers import StableDiffusionPipeline

from .config import Config
from .device_resolver import DeviceResolver


class PipelineLoader:
    """Builds an SD 1.5 pipeline and surfaces text encoder, VAE and UNet.

    A text-to-image diffusion model is three trained networks working
    together:

    * the *text encoder* (CLIP) turns the prompt into token embeddings,
    * the *UNet* denoises a latent image, attending to those embeddings,
    * the *VAE* decodes the final latent into pixels.

    This class loads all three (via diffusers) and keeps them addressable so
    the rest of the package can hook into the UNet's attention layers.
    """

    def __init__(self, config: Config, resolver: DeviceResolver):
        self._config = config
        self._resolver = resolver
        self._pipe = None

    def load(self) -> StableDiffusionPipeline:
        """Download/load weights and move them onto the chosen device."""
        pipe = StableDiffusionPipeline.from_pretrained(
            self._config.model_id,
            torch_dtype=self._resolver.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self._resolver.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    @property
    def pipe(self) -> StableDiffusionPipeline:
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
