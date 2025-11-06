"""Tests for Tessrax ledger normalization orchestration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessrax.core.ledger import ledger_normalizer


def _write_sample_ledger(directory: Path) -> Path:
    ledger_path = directory / "sample.jsonl"
    lines = [
        {
            "actor": "unit-test",
            "event": "create",
            "hash": "abc123",
            "payload": {"value": 1},
            "signature": "sig",
            "timestamp": "2025-11-05T00:00:00Z",
        },
        {
            "actor": "unit-test",
            "event": "update",
            "payload": {"value": 2},
            "timestamp": "2025-11-05T01:00:00Z",
        },
    ]
    with ledger_path.open("w", encoding="utf-8") as handle:
        for item in lines:
            handle.write(json.dumps(item) + "\n")
    return ledger_path


def test_ledger_self_test_produces_receipt(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    _write_sample_ledger(ledger_dir)

    receipt_dir = tmp_path / "receipts"
    result = ledger_normalizer.self_test(target_dir=ledger_dir, receipt_dir=receipt_dir)

    assert result is True
    receipt_path = receipt_dir / "ledger_normalizer_receipt.json"
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["runtime_info"]["records_normalized"] == 2


def test_ledger_self_test_errors_when_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ledger_normalizer.self_test(target_dir=tmp_path)
