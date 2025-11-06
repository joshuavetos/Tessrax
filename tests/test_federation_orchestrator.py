"""Tests for the Tessrax federation orchestrator (v18.6)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from federation.orchestrator import orchestrate_federation


def test_orchestrator_creates_receipt_and_ledger(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    ledger_path = tmp_path / "ledger.jsonl"

    result = asyncio.run(
        orchestrate_federation(
            nodes=3,
            cycles=2,
            seed=99,
            receipt_path=receipt_path,
            ledger_path=ledger_path,
        )
    )

    assert receipt_path.exists()
    assert ledger_path.exists()

    payload = json.loads(receipt_path.read_text())
    assert payload["status"] == "PASS"
    assert payload["runtime_info"]["aggregated_runs"] == 6
    assert payload["runtime_info"]["consensus_root"] == result["runtime_info"]["consensus_root"]

    ledger_lines = [line for line in ledger_path.read_text().splitlines() if line.strip()]
    assert len(ledger_lines) == 2

    distribution = result["runtime_info"]["consensus_distribution"]
    assert isinstance(distribution, dict)
    assert sum(distribution.values()) == result["runtime_info"]["aggregated_runs"]
    assert result["runtime_info"]["consensus_ratio"] > 0
