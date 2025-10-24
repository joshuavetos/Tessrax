from __future__ import annotations

import json
from pathlib import Path

import pytest

from config_loader import (
    ConfigValidationError,
    LoggingConfig,
    TessraxConfig,
    load_config,
    validate_config,
)


def test_validate_config_accepts_valid_schema() -> None:
    payload = {
        "thresholds": {"autonomic": 0.9, "deliberative": 0.4},
        "logging": {"ledger_path": "ledger/custom.jsonl"},
    }
    config = validate_config(payload)
    assert isinstance(config, TessraxConfig)
    assert config.thresholds["autonomic"] == pytest.approx(0.9)
    assert isinstance(config.logging, LoggingConfig)


def test_validate_config_rejects_invalid_threshold() -> None:
    payload = {
        "thresholds": {"autonomic": 2},
        "logging": {"ledger_path": "ledger/output.jsonl"},
    }
    with pytest.raises(ConfigValidationError):
        validate_config(payload)


def test_load_config_reads_from_disk(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "thresholds": {"autonomic": 0.75},
                "logging": {"ledger_path": "ledger/override.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(str(config_path))
    assert config.thresholds["autonomic"] == pytest.approx(0.75)
    assert config.logging.ledger_path == "ledger/override.jsonl"
