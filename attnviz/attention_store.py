"""Accumulate attention maps captured during a denoising run."""

from __future__ import annotations

import math
from typing import Dict, List

import torch


class AttentionStore:
    """Collects per-layer attention maps, averaged over denoising steps.

    The UNet runs an attention layer once per denoising step. Keeping every
    map for every step is memory-heavy and noisy, so this store keeps a
    running mean per layer (a Welford-free incremental average). Maps arrive
    already averaged over attention heads.

    Three buckets are kept, all keyed by the layer's module name and stored on
    CPU as float32:

    * ``cross`` — post-softmax cross-attention probabilities,
    * ``cross_scores`` — pre-softmax cross-attention logits (Q·Kᵀ/√d),
    * ``self`` — post-softmax self-attention probabilities.

    Each bucket tracks its own step counts (cross and cross_scores share layer
    names, so the counts must be separate).
    """

    def __init__(self, self_attn_max_res: int = 32):
        self._self_attn_max_res = self_attn_max_res
        self._cross: Dict[str, torch.Tensor] = {}
        self._cross_scores: Dict[str, torch.Tensor] = {}
        self._self: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, Dict[str, int]] = {"cross": {}, "cross_scores": {}, "self": {}}

    def reset(self) -> None:
        """Drop everything — call before each new generation."""
        self._cross.clear()
        self._cross_scores.clear()
        self._self.clear()
        for counts in self._counts.values():
            counts.clear()

    def record(self, name: str, is_cross: bool, attn: torch.Tensor) -> None:
        """Add one post-softmax map. ``attn`` is [batch, query, key]."""
        if not self._should_keep(is_cross, attn):
            return
        if is_cross:
            self._accumulate("cross", self._cross, name, attn)
        else:
            self._accumulate("self", self._self, name, attn)

    def record_scores(self, name: str, scores: torch.Tensor) -> None:
        """Add one pre-softmax cross-attention logit map [batch, query, key]."""
        self._accumulate("cross_scores", self._cross_scores, name, scores)

    def _should_keep(self, is_cross: bool, attn: torch.Tensor) -> bool:
        if is_cross:
            return True
        res = int(round(math.sqrt(attn.shape[1])))
        return res <= self._self_attn_max_res

    def _accumulate(self, key: str, bucket: Dict[str, torch.Tensor],
                    name: str, tensor: torch.Tensor) -> None:
        m = tensor.detach().to(torch.float32).cpu()
        counts = self._counts[key]
        if name not in bucket:
            bucket[name] = m
            counts[name] = 1
        else:
            n = counts[name]
            bucket[name] = (bucket[name] * n + m) / (n + 1)
            counts[name] = n + 1

    def cross_layers(self) -> Dict[str, torch.Tensor]:
        return dict(self._cross)

    def cross_score_layers(self) -> Dict[str, torch.Tensor]:
        return dict(self._cross_scores)

    def self_layers(self) -> Dict[str, torch.Tensor]:
        return dict(self._self)

    def cross_names_at_res(self, res: int) -> List[str]:
        return [n for n, m in self._cross.items()
                if int(round(math.sqrt(m.shape[1]))) == res]

    def self_names_at_res(self, res: int) -> List[str]:
        return [n for n, m in self._self.items()
                if int(round(math.sqrt(m.shape[1]))) == res]

    def available_self_resolutions(self) -> List[int]:
        res = {int(round(math.sqrt(m.shape[1]))) for m in self._self.values()}
        return sorted(res)
