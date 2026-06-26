"""Load (and switch between) models and run generations for the web app."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import List, Optional

from attnviz import (
    Config,
    CrossAttentionVisualizer,
    DeviceResolver,
    Generator,
    ImageToTextVisualizer,
    PipelineLoader,
    SelfAttentionVisualizer,
)

from .session_store import Session


class ModelService:
    """Owns the currently loaded pipeline and turns prompts into sessions.

    A small registry of allowed checkpoints (same SD architecture, single CLIP
    text encoder) can be switched between at runtime. Switching reloads the
    pipeline; the first use of a checkpoint downloads its weights. Loading and
    generation are serialized with one lock because a single pipeline is not
    safe to run from several requests at once.
    """

    # All use attn1/attn2 cross-attention over CLIP tokens, so the capture +
    # visualization stack works unchanged. SDXL just needs its own pipeline
    # class (two text encoders) — selected via the "kind" field.
    AVAILABLE_MODELS = [
        {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
         "label": "Stable Diffusion 1.5 (512)", "size": 512, "kind": "sd"},
        {"id": "stabilityai/stable-diffusion-xl-base-1.0",
         "label": "Stable Diffusion XL 1.0 (1024)", "size": 1024, "kind": "sdxl"},
    ]

    def __init__(self, config: Config):
        self._base_config = config
        self._resolver = DeviceResolver(config.device, config.dtype)
        self._loader: Optional[PipelineLoader] = None
        self._loaded_model_id: Optional[str] = None
        self._lock = threading.Lock()
        self._progress = {"step": 0, "total": 0, "active": False}
        self._progress_lock = threading.Lock()

    def progress(self) -> dict:
        """Snapshot of the current denoising progress (thread-safe)."""
        with self._progress_lock:
            return dict(self._progress)

    def _set_progress(self, step: int, total: int, active: bool) -> None:
        with self._progress_lock:
            self._progress = {"step": step, "total": total, "active": active}

    @property
    def device_label(self) -> str:
        return f"{self._resolver.device} ({self._resolver.dtype})"

    @property
    def loaded_model_id(self) -> str:
        return self._loaded_model_id or ""

    def available_models(self) -> List[dict]:
        return self.AVAILABLE_MODELS

    def is_allowed(self, model_id: str) -> bool:
        return any(m["id"] == model_id for m in self.AVAILABLE_MODELS)

    def default_model_id(self) -> str:
        return self.AVAILABLE_MODELS[0]["id"]

    def _kind(self, model_id: str) -> str:
        for m in self.AVAILABLE_MODELS:
            if m["id"] == model_id:
                return m.get("kind", "sd")
        return "sd"

    def ensure_model(self, model_id: str) -> bool:
        """Load ``model_id`` if it is not already the active pipeline.

        Returns True if a (re)load actually happened (weights may download).
        """
        with self._lock:
            if model_id == self._loaded_model_id:
                return False
            cfg = replace(self._base_config, model_id=model_id,
                          architecture=self._kind(model_id))
            loader = PipelineLoader(cfg, self._resolver)
            loader.load()
            self._loader = loader
            self._loaded_model_id = model_id
            return True

    def generate(self, model_id: str, prompt: str, steps: int, seed: int,
                 size: int, guidance: float) -> Session:
        """Ensure the right model is loaded, then run one generation."""
        self._set_progress(0, steps, True)
        try:
            self.ensure_model(model_id)
            with self._lock:
                config = replace(
                    self._base_config, model_id=model_id,
                    architecture=self._kind(model_id),
                    num_inference_steps=steps, seed=seed,
                    image_size=size, guidance_scale=guidance,
                )
                on_step = lambda step, total: self._set_progress(step, total, True)
                result = Generator(config, self._loader).generate(prompt, on_step=on_step)
            return self._build_session(config, result)
        finally:
            self._set_progress(steps, steps, False)

    def _build_session(self, config: Config, result) -> Session:
        cross = CrossAttentionVisualizer(result, resolutions=config.cross_attn_resolutions)
        self_attn = SelfAttentionVisualizer(result)
        image2text = ImageToTextVisualizer(cross)
        return Session(result=result, cross=cross, self_attn=self_attn,
                       image2text=image2text)
