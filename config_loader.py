"""Configuration loader for Tessrax-Core."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when the configuration does not match the expected schema."""


_DEFAULT_CONFIG = {
    "thresholds": {"autonomic": 0.8, "deliberative": 0.5},
    "logging": {
        "ledger_path": "ledger/ledger.jsonl",
        "cloud": {
            "enabled": False,
            "provider": "s3",
            "bucket": "tessrax-governance-ledger",
            "region": "us-east-1",
            "endpoint_url": None,
            "use_mock": False,
        },
    },
}


@dataclass(frozen=True)
class CloudLoggingConfig:
    """Configuration for optional cloud logging destinations."""

    enabled: bool
    provider: str
    bucket: str
    region: str
    endpoint_url: str | None
    use_mock: bool


@dataclass(frozen=True)
class LoggingConfig:
    """Logging-related configuration values."""

    ledger_path: str
    cloud: CloudLoggingConfig | None


@dataclass(frozen=True)
class TessraxConfig:
    """Top-level configuration container."""

    thresholds: dict[str, float]
    logging: LoggingConfig


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def _validate_thresholds(values: dict[str, Any]) -> dict[str, float]:
    validated: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(value, (int, float)):
            raise ConfigValidationError(
                f"Threshold '{key}' must be numeric, received {type(value).__name__}"
            )
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ConfigValidationError(
                f"Threshold '{key}' must be in the range [0.0, 1.0]"
            )
        validated[key] = numeric
    return validated


def _validate_cloud_logging(values: dict[str, Any]) -> CloudLoggingConfig:
    enabled = bool(values.get("enabled", False))
    provider = str(values.get("provider", "s3"))
    bucket = str(values.get("bucket", ""))
    region = str(values.get("region", "us-east-1"))
    endpoint = values.get("endpoint_url")
    if endpoint is not None and not isinstance(endpoint, str):
        raise ConfigValidationError("'endpoint_url' must be a string when provided")
    use_mock = bool(values.get("use_mock", False))
    if enabled and not bucket:
        raise ConfigValidationError("Cloud logging requires a non-empty bucket name")
    return CloudLoggingConfig(
        enabled=enabled,
        provider=provider,
        bucket=bucket,
        region=region,
        endpoint_url=endpoint,
        use_mock=use_mock,
    )


def _validate_logging(values: dict[str, Any]) -> LoggingConfig:
    if "ledger_path" not in values:
        raise ConfigValidationError("Logging configuration requires a 'ledger_path'")
    if not isinstance(values["ledger_path"], str):
        raise ConfigValidationError("'ledger_path' must be a string")
    cloud_config = None
    if "cloud" in values and isinstance(values["cloud"], dict):
        cloud_config = _validate_cloud_logging(dict(values["cloud"]))
    return LoggingConfig(ledger_path=values["ledger_path"], cloud=cloud_config)


def validate_config(data: dict[str, Any]) -> TessraxConfig:
    """Validate loaded configuration against the expected schema."""

    if "thresholds" not in data or "logging" not in data:
        raise ConfigValidationError(
            "Configuration must define 'thresholds' and 'logging' sections"
        )
    thresholds = _validate_thresholds(dict(data["thresholds"]))
    logging = _validate_logging(dict(data["logging"]))
    return TessraxConfig(thresholds=thresholds, logging=logging)


def load_config(path: str | None = None) -> TessraxConfig:
    """Load configuration from disk or fall back to defaults."""

    data: dict[str, Any] = dict(_DEFAULT_CONFIG)
    candidate_paths = []
    env_path = os.getenv("TESSRAX_CONFIG_PATH")
    if path:
        candidate_paths.append(path)
    if env_path:
        candidate_paths.append(env_path)
    for candidate_path in candidate_paths:
        candidate = Path(candidate_path)
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
