"""YAML config loading with ``base:`` chaining and recursive merge."""
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config.
    """
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    return _deep_merge(load_config(base), cfg)