"""Tests for the DLK-verified provenance tracer."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tessrax.provenance.tracer import ProvenanceTracer


def test_provenance_record_includes_deterministic_hash_and_timestamp(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []

    def fake_ledger(event: dict[str, object]) -> dict[str, object]:
        events.append(event)
        return event

    key_path = tmp_path / "signing.key"
    receipt_path = tmp_path / "receipt.json"
    tracer = ProvenanceTracer(key_path=key_path, receipt_path=receipt_path, ledger_sink=fake_ledger)

    payload = tracer.record(
        source="Claim: integrity remains above threshold",  # acts as claim text
        agent_id="agent-alpha",
        dataset_hash="abc123",
        reasoning_path="governance::proof",
    )

    expected_hash = hashlib.sha256(
        "Claim: integrity remains above threshold".encode("utf-8")
    ).hexdigest()
    assert payload["claim_hash"] == expected_hash
    assert "timestamp" in payload and payload["timestamp"].endswith("Z") is False
    assert events and events[0]["event_type"] == "PROVENANCE_TRACE"
    assert receipt_path.exists()
