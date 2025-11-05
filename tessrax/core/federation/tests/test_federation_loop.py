"""Integration tests for the Tessrax federated governance loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tessrax.core.federation.network import simulate_cluster
from tessrax.core.federation.node import LEDGER_PATH as DEFAULT_LEDGER_PATH


def _load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_cluster_produces_quorum(tmp_path: Path) -> None:
    ledger = tmp_path / "fed.jsonl"
    result = asyncio.run(simulate_cluster(n=3, rounds=2, ledger_path=ledger))
    assert result is True

    quorum_events = _load_jsonl(ledger)
    assert len(quorum_events) == 2
    assert all(event["event"] == "federated_quorum" for event in quorum_events)
    assert quorum_events[-1]["round"] == 2

    default_events = _load_jsonl(DEFAULT_LEDGER_PATH)
    assert default_events, "Default federated ledger should contain events"
