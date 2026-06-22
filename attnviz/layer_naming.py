"""Short, human-readable names for UNet attention layers."""

from __future__ import annotations

import re

_PATTERN = re.compile(r"(down|mid|up)_block[s]?(?:\.(\d+))?\.attentions\.(\d+)")


def layer_label(name: str, res: int) -> str:
    """Turn a long module path into a compact label, e.g. ``up.1.a2 · 32²``.

    Falls back to the raw name (with resolution) if the path is unexpected.
    """
    match = _PATTERN.search(name)
    if not match:
        return f"{name} · {res}²"
    block, index, attention = match.groups()
    index = f".{index}" if index is not None else ""
    return f"{block}{index}.a{attention} · {res}²"
