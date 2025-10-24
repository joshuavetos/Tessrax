"""Configuration loader for Tessrax-Core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_CONFIG = {
    "thresholds": {"autonomic": 0.8, "deliberative": 0.5},
    "logging": {"ledger_path": "ledger/ledger.jsonl"},
}


@dataclass(frozen=True)
class LoggingConfig:
    """Logging-related configuration values."""

    ledger_path: str


@dataclass(frozen=True)
class TessraxConfig:
    """Top-level configuration container."""

    thresholds: Dict[str, float]
    logging: LoggingConfig


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str] = None) -> TessraxConfig:
    """Load configuration from disk or fall back to defaults."""

    data: Dict[str, Any] = dict(_DEFAULT_CONFIG)
    if path:
        candidate = Path(path)
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            data = _merge_dict(data, loaded)
    logging = LoggingConfig(**data["logging"])
    return TessraxConfig(thresholds=data["thresholds"], logging=logging)


__all__ = ["TessraxConfig", "LoggingConfig", "load_config"]
