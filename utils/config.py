"""YAML config loading with path resolution + shallow merge."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a plain dict."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}")
    return cfg


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `overrides` into a copy of `base`."""
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a value via 'a.b.c' dotted path. Returns `default` if missing."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create every directory listed under paths.* so downstream writes don't fail."""
    for key, value in (cfg.get("paths") or {}).items():
        if not isinstance(value, str):
            continue
        # Heuristic: treat as directory if the key ends in _dir or there's no file suffix
        suffix = Path(value).suffix
        if key.endswith("_dir") or not suffix:
            Path(value).mkdir(parents=True, exist_ok=True)
        else:
            Path(value).parent.mkdir(parents=True, exist_ok=True)
