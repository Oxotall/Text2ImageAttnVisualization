"""A diffusers attention processor that records attention probabilities."""

from __future__ import annotations

import torch

from .attention_store import AttentionStore


class CaptureAttnProcessor:
    """Drop-in replacement for diffusers' ``AttnProcessor`` that also stores
    the attention probabilities it computes.

    Modern diffusers defaults to a fused scaled-dot-product-attention kernel
    that never materializes the attention matrix, so it cannot be inspected.
    This processor reimplements the classic, explicit attention path
    (Q·Kᵀ → softmax → ·V), and on the way it hands the softmax output to an
    :class:`AttentionStore`. The numerical result is identical to the stock
    processor; only the extra bookkeeping differs.

    One instance is bound to one attention module, tagged with that module's
    name and whether it is a cross-attention layer.
    """

    def __init__(self, name: str, is_cross: bool, store: AttentionStore):
        self._name = name
        self._is_cross = is_cross
        self._store = store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, spatial_shape = self._maybe_flatten(attn, hidden_states, temb)
        query, key, value, batch_size, mask = self._project(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )
        probs = attn.get_attention_scores(query, key, mask)
        self._store_probs(probs, batch_size, attn.heads)
        if self._is_cross:
            # also keep the pre-softmax logits (Q·Kᵀ/√d): for text->image the
            # softmax normalizes per-query, so raw scores compare better across
            # image patches for a fixed token.
            self._store_scores(attn, query, key, mask, batch_size)
        out = torch.bmm(probs, value)
        out = attn.batch_to_head_dim(out)
        return self._finalize(attn, out, residual, spatial_shape)

    def _maybe_flatten(self, attn, hidden_states, temb):
        """Conv feature maps come in as [B,C,H,W]; flatten to [B, H*W, C]."""
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        shape = None
        if hidden_states.ndim == 4:
            b, c, h, w = hidden_states.shape
            shape = (b, c, h, w)
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        return hidden_states, shape

    def _project(self, attn, hidden_states, encoder_hidden_states, attention_mask):
        """Compute per-head Q, K, V tensors and the prepared mask."""
        seq_source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, seq_len, _ = seq_source.shape
        mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        return query, key, value, batch_size, mask

    def _store_probs(self, probs: torch.Tensor, batch_size: int, heads: int) -> None:
        """Average over heads and hand [batch, query, key] to the store."""
        q, k = probs.shape[1], probs.shape[2]
        per_batch = probs.reshape(batch_size, heads, q, k).mean(dim=1)
        self._store.record(self._name, self._is_cross, per_batch)

    def _store_scores(self, attn, query, key, mask, batch_size: int) -> None:
        """Recompute Q·Kᵀ/√d (the pre-softmax logits) and store them."""
        scores = torch.baddbmm(
            torch.zeros(1, dtype=query.dtype, device=query.device),
            query, key.transpose(-1, -2), beta=0, alpha=attn.scale,
        )
        if mask is not None:
            scores = scores + mask
        q, k = scores.shape[1], scores.shape[2]
        per_batch = scores.reshape(batch_size, attn.heads, q, k).mean(dim=1)
        self._store.record_scores(self._name, per_batch)

    def _finalize(self, attn, out, residual, spatial_shape):
        """Apply output projection and restore the original tensor shape."""
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if spatial_shape is not None:
            b, c, h, w = spatial_shape
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor
