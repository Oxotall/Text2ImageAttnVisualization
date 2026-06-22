"""Shared helpers for turning attention maps into heatmap overlays."""

from __future__ import annotations

import numpy as np
from PIL import Image


class HeatmapRenderer:
    """Stateless utilities to normalize and overlay attention heatmaps.

    Attention maps live on a small latent grid (e.g. 16x16). To show them on
    top of the generated image they are min-max normalized to [0, 1] and
    resized to the image resolution with bilinear interpolation.
    """

    def normalize(self, grid: np.ndarray) -> np.ndarray:
        """Min-max scale a 2D map into [0, 1]; flat maps become zeros."""
        lo, hi = float(grid.min()), float(grid.max())
        if hi - lo < 1e-12:
            return np.zeros_like(grid)
        return (grid - lo) / (hi - lo)

    def upsample(self, grid: np.ndarray, size: int) -> np.ndarray:
        """Resize a normalized 2D map to ``size`` x ``size`` pixels."""
        img = Image.fromarray((grid * 255).astype(np.uint8))
        img = img.resize((size, size), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def to_overlay(self, grid: np.ndarray, size: int) -> np.ndarray:
        """Normalize then upsample — the common case."""
        return self.upsample(self.normalize(grid), size)

    def blend(self, base: Image.Image, heat: np.ndarray, alpha: float = 0.6,
              cmap=None) -> np.ndarray:
        """Alpha-blend a heatmap (colored via ``cmap``) over the base image."""
        import matplotlib

        cmap = cmap if cmap is not None else matplotlib.colormaps["jet"]
        colored = cmap(heat)[..., :3]
        base_arr = np.asarray(base.convert("RGB"), dtype=np.float32) / 255.0
        blended = (1 - alpha) * base_arr + alpha * colored
        return np.clip(blended, 0.0, 1.0)
