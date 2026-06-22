"""Encode images and heatmap overlays as base64 PNG data URLs."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


class ImageCodec:
    """Turns PIL images and float overlays into ``data:`` URLs for the browser."""

    def pil_to_data_url(self, image: Image.Image) -> str:
        return self._encode(image.convert("RGB"))

    def array_to_data_url(self, array: np.ndarray) -> str:
        """``array`` is a float RGB image in [0, 1] (e.g. a blended overlay)."""
        clipped = np.clip(array, 0.0, 1.0)
        pil = Image.fromarray((clipped * 255).astype(np.uint8))
        return self._encode(pil)

    def _encode(self, pil: Image.Image) -> str:
        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
