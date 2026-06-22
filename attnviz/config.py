"""Configuration for a generation + attention-capture run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    """All knobs for loading the model and running a capture.

    Attributes:
        model_id: HuggingFace repo of the SD 1.5 checkpoint.
        device: "cuda" | "mps" | "cpu" or None to auto-detect.
        dtype: "float16" | "float32" or None to pick a sane default per device.
        image_size: Output image resolution (square), must be a multiple of 8.
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: RNG seed for reproducibility.
        capture_cross: Capture cross-attention (text-token -> image).
        capture_self: Capture self-attention (image -> image).
        self_attn_max_res: Only keep self-attention maps whose spatial side is
            <= this value (self-attention maps grow as res^4, so big maps are
            both memory-hungry and noisy). 32 means up to 32x32 latent grids.
        cross_attn_resolutions: Latent-grid sides to average cross-attention
            over. 16 (i.e. 16x16) is the classic "most semantic" choice.
    """

    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device: Optional[str] = None
    dtype: Optional[str] = None
    image_size: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 0

    capture_cross: bool = True
    capture_self: bool = True
    self_attn_max_res: int = 32
    cross_attn_resolutions: Tuple[int, ...] = field(default_factory=lambda: (16,))

    def latent_size(self) -> int:
        """Latent grid side length (VAE downsamples by 8)."""
        return self.image_size // 8
