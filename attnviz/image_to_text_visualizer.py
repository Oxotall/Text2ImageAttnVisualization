"""Visualize how one image region depends on each input text token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .cross_attention_visualizer import CrossAttentionVisualizer


@dataclass
class TokenWeight:
    """One text token and how strongly a chosen image region attends to it."""

    index: int
    label: str
    weight: float  # min-max normalized across displayed tokens, in [0, 1]
    special: bool = False  # True for the start/end markers


class ImageToTextVisualizer:
    """The transpose of cross-attention: image region -> text tokens.

    Cross-attention is a map [image_position, token]. Fixing a *token* and
    viewing it over image positions gives the text->image heatmap; fixing an
    *image position* and viewing it over tokens (this class) tells you which
    words that patch was "looking at" while being rendered.

    For a clicked pixel it reads that patch's row from the aggregated
    cross-attention grid and min-max normalizes the word-token weights so they
    can color the token chips.
    """

    def __init__(self, cross: CrossAttentionVisualizer):
        self._cross = cross

    def grid_res(self) -> int:
        return self._cross.grid_res

    def available_layers(self):
        """Cross-attention layers that can be inspected separately."""
        return self._cross.available_layers()

    def token_weights(self, x: int, y: int, image_size: int,
                      layer=None, include_special: bool = False) -> List[TokenWeight]:
        """Per-token weights for the patch under pixel ``(x, y)``.

        ``layer`` selects a single UNet cross-attention layer; ``None`` uses
        the average over layers. ``include_special`` decides whether the
        start/end marker tokens participate in the score and its min-max
        normalization. They are excluded by default because they are attention
        sinks that otherwise dominate the scale and flatten the word weights.
        """
        indices = self._indices(include_special)
        grid = self._cross.grid_for(layer)
        col, row = self._pixel_to_cell(x, y, image_size, grid.shape[0])
        vector = grid[row, col]  # [num_tokens]
        normalized = self._min_max([float(vector[i]) for i in indices])
        return [
            TokenWeight(
                index=idx,
                label=self._cross.display_label(idx),
                weight=w,
                special=self._cross.is_special(idx),
            )
            for idx, w in zip(indices, normalized)
        ]

    def _indices(self, include_special: bool) -> List[int]:
        if include_special:
            return self._cross.displayable_token_indices()
        return self._cross.word_token_indices()

    def _pixel_to_cell(self, x: int, y: int, image_size: int, res: int):
        col = min(res - 1, max(0, int(x / image_size * res)))
        row = min(res - 1, max(0, int(y / image_size * res)))
        return col, row

    def _min_max(self, values: List[float]) -> List[float]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]
