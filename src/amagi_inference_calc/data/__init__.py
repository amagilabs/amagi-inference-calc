"""Bundled JSON data (GPUs, models, office presets, calibration)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

try:
    from importlib.resources import files as _files
except ImportError:  # pragma: no cover - Python < 3.9
    _files = None


@lru_cache(maxsize=None)
def load_json(name: str) -> Any:
    if _files is not None:
        text = _files(__name__).joinpath(name).read_text(encoding="utf-8")
    else:  # pragma: no cover
        import pkgutil
        text = pkgutil.get_data(__name__, name).decode("utf-8")
    return json.loads(text)
