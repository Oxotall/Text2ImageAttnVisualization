"""Visualize how image regions depend on input text tokens."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .generator import GenerationResult
from .heatmap_renderer import HeatmapRenderer
from .layer_naming import layer_label


class CrossAttentionVisualizer:
    """Maps every text token to the image regions that attend to it.

    Cross-attention answers "which pixels looked at this word?". For each
    captured layer at a chosen latent resolution, the map is [H*W, num_tokens].

    Two viewing modes are supported:

    * the **average** over the layers at the default resolution (the cleanest,
      most semantic map) — selected with ``layer=None``;
    * a **single UNet layer**, selected by its module name — useful for seeing
      how different depths of the network attend (coarse layers track layout,
      fine layers track edges/detail).
    """

    SPECIAL_TOKENS = ("<|startoftext|>", "<|endoftext|>")
    AGGREGATE = None  # sentinel for "average over layers"
    # source of the values rendered:
    #   "prob"  = post-softmax probabilities (each query's row sums to 1),
    #   "score" = pre-softmax logits Q·Kᵀ/√d (better for text->image).
    PROB = "prob"
    SCORE = "score"

    def __init__(self, result: GenerationResult,
                 resolutions: Optional[Tuple[int, ...]] = None):
        self._result = result
        self._renderer = HeatmapRenderer()
        self._resolutions = resolutions or (16,)
        self._agg_cache: Dict[str, torch.Tensor] = {}
        self._layer_cache: Dict[tuple, torch.Tensor] = {}

    # ---- tokens ---------------------------------------------------------
    def token_labels(self) -> List[str]:
        return self._result.tokens

    def word_token_indices(self) -> List[int]:
        """Indices of real word tokens only (excludes start/end markers)."""
        return [i for i, t in enumerate(self._result.tokens)
                if t not in self.SPECIAL_TOKENS]

    def displayable_token_indices(self) -> List[int]:
        """All token columns worth showing: start, every word, and end."""
        return list(range(len(self._result.tokens)))

    def is_special(self, index: int) -> bool:
        return self._result.tokens[index] in self.SPECIAL_TOKENS

    def display_label(self, index: int) -> str:
        """Human-friendly chip label, with readable names for the markers."""
        token = self._result.tokens[index]
        if token == "<|startoftext|>":
            return "[start]"
        if token == "<|endoftext|>":
            return "[end]"
        return token

    # ---- layers ---------------------------------------------------------
    def available_layers(self) -> List[Dict]:
        """List of capturable cross-attention layers (UNet order preserved)."""
        layers = []
        for name, m in self._result.store.cross_layers().items():
            res = int(round(math.sqrt(m.shape[1])))
            layers.append({"name": name, "label": layer_label(name, res),
                           "res": res})
        return layers

    def grid_for(self, layer: Optional[str], source: str = PROB) -> torch.Tensor:
        """[res, res, num_tokens] grid for one layer (or the average).

        ``source`` is ``"prob"`` (post-softmax) or ``"score"`` (pre-softmax).
        """
        if layer is self.AGGREGATE:
            return self._aggregate(source)
        key = (source, layer)
        if key not in self._layer_cache:
            self._layer_cache[key] = self._build_layer_grid(layer, source)
        return self._layer_cache[key]

    def grid_res_for(self, layer: Optional[str], source: str = PROB) -> int:
        return self.grid_for(layer, source).shape[0]

    # ---- backward-compatible aggregate accessors (post-softmax) ---------
    @property
    def grid(self) -> torch.Tensor:
        return self._aggregate(self.PROB)

    @property
    def grid_res(self) -> int:
        return self._aggregate(self.PROB).shape[0]

    # ---- rendering ------------------------------------------------------
    def token_heatmap(self, token_index: int, layer: Optional[str] = None,
                      source: str = SCORE) -> np.ndarray:
        """Normalized, image-sized heatmap for one token column.

        Defaults to pre-softmax ``"score"``: for text->image the softmax is
        taken per query patch, so raw logits compare better across patches.
        """
        grid = self.grid_for(layer, source)[..., token_index].numpy()
        return self._renderer.to_overlay(grid, self._image_size())

    def overlay(self, token_index: int, alpha: float = 0.6,
                layer: Optional[str] = None, source: str = SCORE) -> np.ndarray:
        heat = self.token_heatmap(token_index, layer=layer, source=source)
        return self._renderer.blend(self._result.image, heat, alpha=alpha)

    def all_word_heatmaps(self) -> Dict[str, np.ndarray]:
        """label -> heatmap for every real word token (averaged)."""
        out = {}
        for i in self.word_token_indices():
            out[f"{i}:{self._result.tokens[i]}"] = self.token_heatmap(i)
        return out

    # ---- internals ------------------------------------------------------
    def _maps(self, source: str) -> Dict[str, torch.Tensor]:
        store = self._result.store
        return store.cross_score_layers() if source == self.SCORE else store.cross_layers()

    def _build_layer_grid(self, layer: str, source: str) -> torch.Tensor:
        layers = self._maps(source)
        if layer not in layers:
            raise KeyError(f"Unknown cross-attention layer: {layer}")
        m = layers[layer][self._result.cond_batch_index]  # [hw, num_tokens]
        res = int(round(math.sqrt(m.shape[0])))
        return m.reshape(res, res, m.shape[-1])

    def _aggregate(self, source: str) -> torch.Tensor:
        """Average cross-attention maps over the chosen resolutions/layers."""
        if source not in self._agg_cache:
            collected = self._collect_at(max(self._resolutions), source)
            if not collected:
                collected = self._collect_best_available(source)
            stacked = torch.stack(collected, dim=0).mean(dim=0)
            res = int(round(math.sqrt(stacked.shape[0])))
            self._agg_cache[source] = stacked.reshape(res, res, stacked.shape[-1])
        return self._agg_cache[source]

    def _collect_at(self, res: int, source: str) -> List[torch.Tensor]:
        idx = self._result.cond_batch_index
        maps = self._maps(source)
        return [maps[n][idx] for n in self._result.store.cross_names_at_res(res)]

    def _collect_best_available(self, source: str) -> List[torch.Tensor]:
        idx = self._result.cond_batch_index
        layers = self._maps(source)
        if not layers:
            raise RuntimeError("No cross-attention was captured.")
        by_res: Dict[int, List[torch.Tensor]] = {}
        for name, m in layers.items():
            res = int(round(math.sqrt(m.shape[1])))
            by_res.setdefault(res, []).append(m[idx])
        return max(by_res.values(), key=len)

    def _image_size(self) -> int:
        return self._result.image.size[0]
