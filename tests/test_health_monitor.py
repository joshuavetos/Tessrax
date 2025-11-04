"""Tests for the auto-governance health monitor."""

from __future__ import annotations

import json
from pathlib import Path

import tessrax.diagnostics.health_monitor as health_monitor


def test_run_health_check_appends_ledger_and_writes_summary(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    original_append = health_monitor.ledger_append
    health_monitor.ledger_append = lambda payload: events.append(payload)
    try:
        summary_path = tmp_path / "health_summary.json"
        result = health_monitor.run_health_check(
            outcomes=[0.9, 0.92, 0.95], summary_path=summary_path
        )
    finally:
        health_monitor.ledger_append = original_append

    assert summary_path.exists()
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert events and events[0]["event"] == "HEALTH_CHECK"
    assert result["integrity"] == summary_data["integrity"]
