"""Metabolic reconciliation engine."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from ..audit import AuditKernel
from ..ledger import Ledger
from ..schema import ClarityStatement
from ..types import Claim, ContradictionRecord


@dataclass
class _ClarityDecision:
    """Proxy decision used to anchor clarity statements in the ledger."""

    statement: ClarityStatement

    def to_summary(self) -> dict:
        return self.statement.to_receipt()


class ReconciliationEngine:
    """Generate clarity statements from contradictions and log them to the ledger."""

    def __init__(self, audit_kernel: AuditKernel, ledger: Optional[Ledger] = None) -> None:
        self.audit_kernel = audit_kernel
        self.ledger = ledger or Ledger()

    def reconcile(self, contradictions: Sequence[ContradictionRecord]) -> List[ClarityStatement]:
        """Reconcile a batch of contradictions into clarity statements."""

        statements: List[ClarityStatement] = []
        for record in contradictions:
            insight = self.audit_kernel.assess(record)
            clarity_fuel = insight.confidence * 10.0
            statement = ClarityStatement(
                subject=record.claim_a.subject,
                metric=record.claim_a.metric,
                contradiction_ids=[record.claim_a.claim_id, record.claim_b.claim_id],
                synthesis=insight.narrative,
                confidence=insight.confidence,
                severity=record.severity,
                clarity_fuel=clarity_fuel,
                rationale=insight.narrative,
            )
            statements.append(statement)
            self.ledger.append(_ClarityDecision(statement))
        return statements


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _claim_from_payload(payload: dict) -> Claim:
    timestamp_raw = payload.get("timestamp")
    if timestamp_raw:
        timestamp = _parse_timestamp(timestamp_raw)
    else:
        timestamp = datetime.now(timezone.utc)
    return Claim(
        claim_id=payload.get("claim_id", "unknown"),
        subject=payload.get("subject", "unspecified"),
        metric=payload.get("metric", "unspecified"),
        value=float(payload.get("value", 0.0)),
        unit=payload.get("unit", ""),
        timestamp=timestamp,
        source=payload.get("source", "unknown"),
        context=dict(payload.get("context", {})),
    )


def _record_from_payload(payload: dict) -> ContradictionRecord:
    return ContradictionRecord(
        claim_a=_claim_from_payload(payload["claim_a"]),
        claim_b=_claim_from_payload(payload["claim_b"]),
        severity=payload.get("severity", "medium"),
        delta=float(payload.get("delta", 0.0)),
        reasoning=payload.get("reasoning", "No rationale supplied."),
    )


def load_contradictions(path: Path) -> List[ContradictionRecord]:
    """Load contradiction records from a JSON file."""

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):  # pragma: no cover - defensive
        raise ValueError("Expected list of contradiction records")
    return [_record_from_payload(item) for item in data]


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconcile contradictions into clarity statements")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a JSON file containing contradiction records",
    )
    parser.add_argument(
        "--export-ledger",
        dest="export_ledger",
        type=Path,
        help="Optional path to write the generated ledger receipts as JSONL",
    )
    return parser


def _emit_ledger(ledger: Ledger, path: Path) -> None:
    ledger.export(path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_cli()
    args = parser.parse_args(argv)

    contradictions = load_contradictions(args.input)
    engine = ReconciliationEngine(AuditKernel())
    statements = engine.reconcile(contradictions)

    for statement in statements:
        print(json.dumps(statement.to_receipt(), indent=2, sort_keys=True))

    if args.export_ledger:
        _emit_ledger(engine.ledger, args.export_ledger)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
