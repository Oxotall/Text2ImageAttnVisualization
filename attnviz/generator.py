"""Run a generation while capturing attention, returning image + maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from PIL import Image

from .attention_controller import AttentionController
from .attention_store import AttentionStore
from .config import Config
from .pipeline_loader import PipelineLoader
from .token_decoder import TokenDecoder


@dataclass
class GenerationResult:
    """Everything a visualizer needs from one run."""

    image: Image.Image
    prompt: str
    tokens: List[str]
    store: AttentionStore
    latent_size: int
    # Whether classifier-free guidance doubled the batch (uncond first, cond
    # second). When True, visualizers read batch index 1 (the conditional).
    cond_batch_index: int


class Generator:
    """Generates an image and records the UNet's attention along the way.

    It plugs an :class:`AttentionController` into the loaded UNet, runs the
    standard diffusers sampling loop (the capture processors fill the store
    transparently), and bundles the resulting image with the captured maps.
    """

    def __init__(self, config: Config, loader: PipelineLoader):
        self._config = config
        self._loader = loader
        self._store = AttentionStore(self_attn_max_res=config.self_attn_max_res)
        self._decoder = TokenDecoder(loader.tokenizer)

    def generate(self, prompt: str, on_step=None) -> GenerationResult:
        """Produce an image for ``prompt`` and capture its attention maps.

        ``on_step(step, total)`` is called once per denoising step (1-based) so
        callers can show a progress bar.
        """
        self._store.reset()
        controller = AttentionController(
            self._loader.unet, self._store,
            capture_cross=self._config.capture_cross,
            capture_self=self._config.capture_self,
        )
        with controller:
            image = self._run_pipe(prompt, on_step)
        return self._build_result(prompt, image)

    def _run_pipe(self, prompt: str, on_step=None) -> Image.Image:
        generator = torch.Generator(device="cpu").manual_seed(self._config.seed)
        kwargs = dict(
            prompt=prompt,
            height=self._config.image_size,
            width=self._config.image_size,
            num_inference_steps=self._config.num_inference_steps,
            guidance_scale=self._config.guidance_scale,
            generator=generator,
        )
        if on_step is not None:
            total = self._config.num_inference_steps

            def _callback(_pipe, step_index, _timestep, callback_kwargs):
                on_step(step_index + 1, total)
                return callback_kwargs

            kwargs["callback_on_step_end"] = _callback
        output = self._loader.pipe(**kwargs)
        return output.images[0]

    def _build_result(self, prompt: str, image: Image.Image) -> GenerationResult:
        cond_index = 1 if self._config.guidance_scale > 1.0 else 0
        return GenerationResult(
            image=image,
            prompt=prompt,
            tokens=self._decoder.tokens(prompt),
            store=self._store,
            latent_size=self._config.latent_size(),
            cond_batch_index=cond_index,
        )

    @property
    def decoder(self) -> TokenDecoder:
        return self._decoder
