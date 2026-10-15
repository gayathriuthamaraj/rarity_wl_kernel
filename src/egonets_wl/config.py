"""Small YAML configuration loader."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping or raise a clear error for malformed configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must contain a YAML mapping: {path}")
    return value
