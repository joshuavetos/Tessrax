"""Tests for the Tessrax drift visualizer dashboard (v18.3)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from dashboard import drift_visualizer
from tessrax.core.ledger import Ledger


@pytest.fixture()
def seeded_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary ledger populated with integrity status events."""

    ledger_path = tmp_path / "telemetry.jsonl"
    ledger = Ledger(ledger_path)
    base = datetime.now(timezone.utc)
    entries = []
    for index in range(3):
        entries.append(
            {
                "directive": "INTEGRITY_STATUS",
                "timestamp": (base + timedelta(seconds=index)).isoformat().replace("+00:00", "Z"),
                "entropy": 0.5 + index * 0.1,
                "drift": 0.02 * index,
            }
        )
    ledger.append_many(entries)
    monkeypatch.setattr(drift_visualizer, "_ledger_path", lambda: ledger_path)
    return ledger_path


def test_load_metrics_returns_sorted_dataframe(seeded_ledger: Path) -> None:
    """``load_metrics`` returns a chronologically sorted dataframe."""

    frame = drift_visualizer.load_metrics()
    assert list(frame.columns) == ["entropy", "drift"]
    assert frame.index.is_monotonic_increasing
    assert pytest.approx(frame.iloc[-1]["entropy"], rel=1e-6) == 0.7


def test_emit_receipt_writes_governance_payload(tmp_path: Path) -> None:
    """Receipt emission honours governance metadata and hashing."""

    frame = pd.DataFrame({"entropy": [0.1], "drift": [0.01]}, index=pd.DatetimeIndex([datetime.now(timezone.utc)], name="timestamp"))
    receipt_path = drift_visualizer._emit_receipt(frame, out_dir=tmp_path)
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["directive"] == "SAFEPOINT_VISUAL_REPAIR_V18_3"
    assert data["integrity"] >= 0.95
    assert data["legitimacy"] >= 0.9
    digest_source = json.dumps({k: v for k, v in data.items() if k != "hash"}, sort_keys=True).encode("utf-8")
    assert data["hash"] == hashlib.sha256(digest_source).hexdigest()
