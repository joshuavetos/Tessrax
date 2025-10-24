"""Configuration loader for Tessrax-Core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigValidationError(ValueError):
    """Raised when the configuration does not match the expected schema."""


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


def _validate_thresholds(values: Dict[str, Any]) -> Dict[str, float]:
    validated: Dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(f"Threshold '{key}' must be numeric, received {type(value).__name__}")
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ConfigValidationError(f"Threshold '{key}' must be in the range [0.0, 1.0]")
        validated[key] = numeric
    return validated


def _validate_logging(values: Dict[str, Any]) -> LoggingConfig:
    if "ledger_path" not in values:
        raise ConfigValidationError("Logging configuration requires a 'ledger_path'")
    if not isinstance(values["ledger_path"], str):
        raise ConfigValidationError("'ledger_path' must be a string")
    return LoggingConfig(ledger_path=values["ledger_path"])


def validate_config(data: Dict[str, Any]) -> TessraxConfig:
    """Validate loaded configuration against the expected schema."""

    if "thresholds" not in data or "logging" not in data:
        raise ConfigValidationError("Configuration must define 'thresholds' and 'logging' sections")
    thresholds = _validate_thresholds(dict(data["thresholds"]))
    logging = _validate_logging(dict(data["logging"]))
    return TessraxConfig(thresholds=thresholds, logging=logging)


def load_config(path: Optional[str] = None) -> TessraxConfig:
    """Load configuration from disk or fall back to defaults."""

    data: Dict[str, Any] = dict(_DEFAULT_CONFIG)
    if path:
        candidate = Path(path)
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            data = _merge_dict(data, loaded)
    return validate_config(data)


__all__ = [
    "TessraxConfig",
    "LoggingConfig",
    "ConfigValidationError",
    "validate_config",
    "load_config",
]
