"""Visualize how image regions depend on other image regions."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from .generator import GenerationResult
from .heatmap_renderer import HeatmapRenderer
from .layer_naming import layer_label


class SelfAttentionVisualizer:
    """Maps a chosen image region to the other regions it attends to.

    Self-attention answers "when the model rendered this patch, which other
    patches did it look at?". For a layer at latent resolution R, the map is
    [R*R, R*R]; row q is the distribution over keys for query patch q.
    Picking a patch (from a pixel location) and reshaping its row to the latent
    grid yields a spatial heatmap of what that patch depended on.

    Like cross-attention, two viewing modes are supported: the **average** over
    the layers at the default resolution (``layer=None``), or a **single UNet
    self-attention layer** selected by its module name.
    """

    AGGREGATE = None  # sentinel for "average over layers"

    def __init__(self, result: GenerationResult, resolution: Optional[int] = None):
        self._result = result
        self._renderer = HeatmapRenderer()
        self._resolution = resolution or self._default_resolution()
        self._aggregated = self._aggregate()  # [R*R, R*R]
        self._res = int(round(math.sqrt(self._aggregated.shape[0])))
        self._layer_cache: Dict[str, torch.Tensor] = {}

    @property
    def resolution(self) -> int:
        return self._res

    def available_resolutions(self) -> List[int]:
        return self._result.store.available_self_resolutions()

    def available_layers(self) -> List[Dict]:
        """Capturable self-attention layers (those within the resolution cap)."""
        layers = []
        for name, m in self._result.store.self_layers().items():
            res = int(round(math.sqrt(m.shape[1])))
            layers.append({"name": name, "label": layer_label(name, res),
                           "res": res})
        return layers

    def grid_res_for(self, layer: Optional[str] = None) -> int:
        """Latent side length of the chosen layer's grid (or the average)."""
        return self._grid_and_res(layer)[1]

    def region_heatmap(self, x: int, y: int,
                       layer: Optional[str] = None) -> np.ndarray:
        """Heatmap of what the patch under pixel (x, y) attended to."""
        grid, res = self._grid_and_res(layer)
        query_index = self._pixel_to_query(x, y, res)
        row = grid[query_index].reshape(res, res).numpy()
        return self._renderer.to_overlay(row, self._image_size())

    def region_heatmap_by_cell(self, col: int, row: int,
                               layer: Optional[str] = None) -> np.ndarray:
        """Same as :meth:`region_heatmap` but indexed by latent grid cell."""
        grid, res = self._grid_and_res(layer)
        col = min(res - 1, max(0, col))
        row = min(res - 1, max(0, row))
        query_index = row * res + col
        data = grid[query_index].reshape(res, res).numpy()
        return self._renderer.to_overlay(data, self._image_size())

    def overlay(self, x: int, y: int, alpha: float = 0.6,
                layer: Optional[str] = None) -> np.ndarray:
        heat = self.region_heatmap(x, y, layer=layer)
        return self._renderer.blend(self._result.image, heat, alpha=alpha)

    def _grid_and_res(self, layer: Optional[str]):
        if layer is self.AGGREGATE:
            return self._aggregated, self._res
        grid = self._layer_grid(layer)
        return grid, int(round(math.sqrt(grid.shape[0])))

    def _layer_grid(self, layer: str) -> torch.Tensor:
        if layer not in self._layer_cache:
            store = self._result.store
            layers = store.self_layers()
            if layer not in layers:
                raise KeyError(f"Unknown self-attention layer: {layer}")
            self._layer_cache[layer] = layers[layer][self._result.cond_batch_index]
        return self._layer_cache[layer]

    def _pixel_to_query(self, x: int, y: int, res: int) -> int:
        size = self._image_size()
        col = min(res - 1, max(0, int(x / size * res)))
        row = min(res - 1, max(0, int(y / size * res)))
        return row * res + col

    def _default_resolution(self) -> int:
        available = self._result.store.available_self_resolutions()
        if not available:
            raise RuntimeError("No self-attention was captured.")
        return available[-1]  # largest captured grid — most spatial detail

    def _aggregate(self) -> torch.Tensor:
        store = self._result.store
        idx = self._result.cond_batch_index
        names = store.self_names_at_res(self._resolution)
        if not names:
            raise RuntimeError(f"No self-attention captured at res {self._resolution}.")
        maps = [store.self_layers()[n][idx] for n in names]
        return torch.stack(maps, dim=0).mean(dim=0)

    def _image_size(self) -> int:
        return self._result.image.size[0]
