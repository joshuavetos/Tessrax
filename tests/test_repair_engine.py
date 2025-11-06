"""Tests for the Tessrax autonomous repair engine (v18.4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tessrax.core.ledger import Ledger
from tessrax.core.repair_engine import RepairEngine, emit_repair_receipt


@pytest.fixture()
def prepared_ledger(tmp_path: Path) -> Ledger:
    """Populate a ledger with integrity status entries requiring repair."""

    ledger_path = tmp_path / "repairs.jsonl"
    ledger = Ledger(ledger_path)
    ledger.append(
        {
            "directive": "INTEGRITY_STATUS",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entropy": 0.92,
            "drift": 0.05,
            "action": "KEY_ROTATION_TRIGGERED",
            "rotation_receipt": {"hash": "cafebabe"},
        }
    )
    return ledger


def test_repair_engine_records_receipt(prepared_ledger: Ledger, tmp_path: Path) -> None:
    """Autonomous repair emits clarity and repair receipts."""

    engine = RepairEngine(prepared_ledger)
    repairs = engine.evaluate_once()
    assert repairs, "Expected at least one repair to be recorded"
    ledger_payloads = [entry["payload"] for entry in prepared_ledger.receipts()]
    repair_entries = [payload for payload in ledger_payloads if payload["directive"] == "REPAIR_RECEIPT"]
    assert repair_entries, "Repair receipt missing from ledger"
    receipt_path = emit_repair_receipt(repairs, out_dir=tmp_path)
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["directive"] == "SAFEPOINT_VISUAL_REPAIR_V18_4"
    assert data["integrity"] >= 0.95
    assert data["events"], "Expected repair events to be recorded"
