"""Config loading and logging setup."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (default: configs/default.yaml)."""
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
