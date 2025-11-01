from __future__ import annotations

import json
from datetime import datetime, timezone

from tessrax.audit import AuditKernel
from tessrax.metabolism import reconcile
from tessrax.metabolism.reconcile import ReconciliationEngine
from tessrax.types import Claim, ContradictionRecord


def _claim(claim_id: str, subject: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        subject=subject,
        metric="oxygen",
        value=21.0,
        unit="%",
        timestamp=datetime.now(timezone.utc),
        source="sensor",
        context={"region": "habitat"},
    )


def test_reconciliation_engine_emits_clarity_receipt(tmp_path) -> None:
    claim_a = _claim("claim-a", "biosphere")
    claim_b = _claim("claim-b", "biosphere")
    record = ContradictionRecord(
        claim_a=claim_a,
        claim_b=claim_b,
        severity="high",
        delta=0.18,
        reasoning="Sensors disagree on oxygenation levels.",
    )

    audit_log = tmp_path / "audit.jsonl"
    diagnostics_log = tmp_path / "diag.jsonl"
    engine = ReconciliationEngine(
        AuditKernel(),
        audit_log_path=audit_log,
        diagnostics_path=diagnostics_log,
        engine_seed=42,
    )
    statements = engine.reconcile([record])

    assert len(statements) == 1
    statement = statements[0]
    assert statement.event_type == "CLARITY_GENERATION"
    assert 0.0 <= statement.confidence <= 1.0

    receipts = engine.ledger.receipts()
    assert len(receipts) == 1
    payload = receipts[0].to_json()

    # Validate JSON serialisation and event typing.
    json.dumps(payload)
    assert payload["event_type"] == "CLARITY_GENERATION"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["severity"] == "high"
    assert payload["action"] == "SYNTHESIZE_CLARITY"
    assert payload["rationale"]

    audit_entries = [
        json.loads(line)
        for line in audit_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert audit_entries
    assert audit_entries[0]["engine_seed"] == 42
    assert audit_entries[0]["ordered_inputs"] == ["claim-a", "claim-b"]

    diagnostics_entries = [
        json.loads(line)
        for line in diagnostics_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert diagnostics_entries
    assert diagnostics_entries[0]["processed_events"] == 1
    assert diagnostics_entries[0]["input_count"] == 1


def test_load_and_cli_flow(tmp_path, capsys) -> None:
    payload = [
        {
            "claim_a": {
                "claim_id": "claim-x",
                "subject": "biosphere",
                "metric": "oxygen",
                "value": 20.0,
                "unit": "%",
                "timestamp": "2025-01-01T00:00:00Z",
                "source": "sensor-a",
            },
            "claim_b": {
                "claim_id": "claim-y",
                "subject": "biosphere",
                "metric": "oxygen",
                "value": 22.0,
                "unit": "%",
                "timestamp": "2025-01-01T00:01:00Z",
                "source": "sensor-b",
            },
            "severity": "medium",
            "delta": 0.15,
            "reasoning": "Gradual drift detected between redundant sensors.",
        }
    ]

    input_path = tmp_path / "contradictions.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"

    records = reconcile.load_contradictions(input_path)
    assert len(records) == 1
    assert records[0].claim_a.claim_id == "claim-x"

    parser = reconcile.build_cli()
    parsed = parser.parse_args([str(input_path), "--export-ledger", str(ledger_path)])
    assert parsed.export_ledger == ledger_path

    reconcile.main([str(input_path), "--export-ledger", str(ledger_path)])
    captured = capsys.readouterr()
    rendered = json.loads(captured.out)
    assert rendered["event_type"] == "CLARITY_GENERATION"
    assert rendered["action"] == "SYNTHESIZE_CLARITY"
    assert rendered["severity"] == "medium"
    assert ledger_path.exists()
