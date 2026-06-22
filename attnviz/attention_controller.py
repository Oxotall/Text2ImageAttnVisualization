"""Install capture processors on every UNet attention layer."""

from __future__ import annotations

from .attention_store import AttentionStore
from .capture_attn_processor import CaptureAttnProcessor


class AttentionController:
    """Swaps the UNet's attention processors for capturing ones.

    In the SD 1.5 UNet, attention layers are named with a ``attn1`` suffix for
    self-attention and ``attn2`` for cross-attention. This controller walks
    the processor dict, replaces each with a :class:`CaptureAttnProcessor`
    tagged by name and type, and can later restore the originals.
    """

    def __init__(self, unet, store: AttentionStore,
                 capture_cross: bool = True, capture_self: bool = True):
        self._unet = unet
        self._store = store
        self._capture_cross = capture_cross
        self._capture_self = capture_self
        self._original = None

    def install(self) -> None:
        """Replace attention processors with capturing ones."""
        self._original = dict(self._unet.attn_processors)
        new_processors = {}
        for name, original in self._unet.attn_processors.items():
            new_processors[name] = self._build_processor(name, original)
        self._unet.set_attn_processor(new_processors)

    def restore(self) -> None:
        """Put the model's original processors back."""
        if self._original is not None:
            self._unet.set_attn_processor(self._original)
            self._original = None

    def _build_processor(self, name: str, original):
        is_cross = name.endswith("attn2.processor")
        wanted = (is_cross and self._capture_cross) or (not is_cross and self._capture_self)
        if not wanted:
            return original
        return CaptureAttnProcessor(name, is_cross, self._store)

    def __enter__(self) -> "AttentionController":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()
