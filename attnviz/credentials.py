"""Optional HuggingFace credentials loading — gracefully absent.

Reads an HF_TOKEN from a local ``.env`` file (if one exists) and exports it to
the environment so huggingface_hub / diffusers pick it up automatically when
downloading gated models (e.g. SDXL, SD 2.1). If the file is missing or the
token is blank, this is a silent no-op and the app still runs — ungated models
like SD 1.5 work with no token at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_CANDIDATE_NAMES = (".env", "credentials.env")


def load_hf_token(explicit_path: Optional[str] = None) -> Optional[str]:
    """Load HF_TOKEN from a credentials file if present and export it.

    A token already set in the shell environment always wins. Returns the
    effective token, or None if there is none (which is fine).
    """
    path = _find_file(explicit_path)
    token = _parse_token(path) if path else None
    if token:
        # setdefault → an existing shell HF_TOKEN takes precedence
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return os.environ.get("HF_TOKEN")


def _find_file(explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.is_file() else None
    roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for root in roots:
        for name in _CANDIDATE_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def _parse_token(path: Path) -> Optional[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'") or None
    return None
