"""Render captured attention into static PNG figures (for the CLI)."""

from __future__ import annotations

import os
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless backend for CLI use
import matplotlib.pyplot as plt  # noqa: E402

from .cross_attention_visualizer import CrossAttentionVisualizer  # noqa: E402
from .generator import GenerationResult  # noqa: E402
from .self_attention_visualizer import SelfAttentionVisualizer  # noqa: E402


class FigureSaver:
    """Writes the generated image and attention grids to an output folder."""

    def __init__(self, out_dir: str):
        self._out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_all(self, result: GenerationResult) -> List[str]:
        """Save image + cross-attention grid + self-attention grid."""
        paths = [self._save_image(result)]
        paths.append(self._save_cross_grid(result))
        paths.append(self._save_self_grid(result))
        return [p for p in paths if p]

    def _save_image(self, result: GenerationResult) -> str:
        path = os.path.join(self._out_dir, "generated.png")
        result.image.save(path)
        return path

    def _save_cross_grid(self, result: GenerationResult) -> str:
        viz = CrossAttentionVisualizer(result)
        indices = viz.displayable_token_indices()
        if not indices:
            return ""
        fig, axes = self._make_axes(len(indices) + 1)
        self._show(axes[0], result.image, "input")
        for ax, i in zip(axes[1:], indices):
            self._show(ax, viz.overlay(i), viz.display_label(i))
        path = os.path.join(self._out_dir, "cross_attention.png")
        self._finish(fig, path)
        return path

    def _save_self_grid(self, result: GenerationResult) -> str:
        try:
            viz = SelfAttentionVisualizer(result)
        except RuntimeError:
            return ""
        size = result.image.size[0]
        points = self._sample_points(size)
        fig, axes = self._make_axes(len(points) + 1)
        self._show(axes[0], result.image, "input")
        for ax, (x, y) in zip(axes[1:], points):
            self._show(ax, viz.overlay(x, y), f"({x},{y})")
            ax.scatter([x], [y], c="white", s=40, edgecolors="black")
        path = os.path.join(self._out_dir, "self_attention.png")
        self._finish(fig, path)
        return path

    def _sample_points(self, size: int):
        q = size // 4
        return [(q, q), (3 * q, q), (size // 2, size // 2), (q, 3 * q), (3 * q, 3 * q)]

    def _make_axes(self, n: int):
        cols = min(6, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for extra in axes[n:]:
            extra.axis("off")
        return fig, list(axes)

    def _show(self, ax, image, title: str) -> None:
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def _finish(self, fig, path: str) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
