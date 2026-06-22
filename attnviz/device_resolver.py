"""Resolve compute device and dtype, auto-detecting when not specified."""

from __future__ import annotations

from typing import Optional

import torch


class DeviceResolver:
    """Picks a torch device and dtype, honoring explicit overrides.

    Auto-detection order is CUDA > MPS (Apple Silicon) > CPU. The default
    dtype is float16 on CUDA (fast, half the memory) and float32 everywhere
    else (float16 is unreliable on MPS/CPU for diffusion).
    """

    def __init__(self, device: Optional[str] = None, dtype: Optional[str] = None):
        self._device = self._resolve_device(device)
        self._dtype = self._resolve_dtype(dtype)

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _resolve_device(self, device: Optional[str]) -> str:
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, dtype: Optional[str]) -> torch.dtype:
        if dtype:
            return torch.float16 if dtype == "float16" else torch.float32
        return torch.float16 if self._device == "cuda" else torch.float32

    def __repr__(self) -> str:
        return f"DeviceResolver(device={self._device!r}, dtype={self._dtype})"
