"""Metabolic reconciliation engine."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from tessrax.audit import AuditKernel
from tessrax.governance import route_to_governance_lane
from tessrax.ledger import Ledger
from tessrax.physics.dynamics import SystemDynamics
from tessrax.physics.phases import GovernancePhase, PhaseTransition
from tessrax.schema import ClarityStatement
from tessrax.types import Claim, ContradictionRecord


class DriftTracker:
    """Tracks and stabilises epistemic drift across reconciliation cycles."""

    def __init__(self) -> None:
        self.history: list[tuple[str, float]] = []

    def update(self, integrity_score: float, severity: float = 1.0) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        weighted_score = integrity_score * severity
        self.history.append((timestamp, weighted_score))
        if len(self.history) > 100:
            self.history.pop(0)

    def drift(self) -> float:
        if len(self.history) < 2:
            return 0.0
        scores = [score for _, score in self.history]
        baseline = sum(scores[:-1]) / max(len(scores) - 1, 1)
        return abs(scores[-1] - baseline)


@dataclass
class _ClarityDecision:
    """Proxy decision used to anchor clarity statements in the ledger."""

    statement: ClarityStatement

    def to_summary(self) -> dict:
        return self.statement.to_receipt()

    def to_decision_payload(self) -> dict:
        receipt = self.statement.to_receipt()
        contradiction = {
            "contradiction_ids": receipt.get("contradiction_ids", []),
            "severity": receipt.get("severity"),
            "subject": receipt.get("subject"),
            "metric": receipt.get("metric"),
        }
        return {
            "protocol": "metabolism",
            "issued_at": receipt.get("timestamp"),
            "clarity_fuel": receipt.get("clarity_fuel"),
            "confidence": receipt.get("confidence"),
            "synthesis": receipt.get("synthesis"),
            "rationale": receipt.get("rationale"),
            "contradiction": contradiction,
        }


class ReconciliationEngine:
    """Generate clarity statements from contradictions and log them to the ledger."""

    def __init__(
        self,
        audit_kernel: AuditKernel,
        ledger: Ledger | None = None,
        *,
        audit_log_path: Path | None = None,
        diagnostics_path: Path | None = None,
        engine_seed: int | None = None,
    ) -> None:
        self.audit_kernel = audit_kernel
        self.ledger = ledger or Ledger()
        self._audit_log_path = (
            Path(audit_log_path)
            if audit_log_path
            else Path("logs/metabolism_audit.jsonl")
        )
        self._diagnostics_path = (
            Path(diagnostics_path)
            if diagnostics_path
            else Path("logs/metabolism_diagnostics.jsonl")
        )
        self._engine_seed = engine_seed if engine_seed is not None else 0
        self._drift_tracker = DriftTracker()
        self._clarity_baseline = 0.5
        self.phase_transition = PhaseTransition()
        self.dynamics = SystemDynamics()
        self._current_phase = GovernancePhase.STABLE

    @staticmethod
    def _kalman_update(
        prior_mean: float,
        measurement: float,
        measurement_confidence: float,
        process_uncertainty: float = 0.05,
    ) -> float:
        """Blend prior clarity belief with new measurement using a Kalman gain."""

        meas_noise = max(1.0 - measurement_confidence, 1e-6)
        K_t = process_uncertainty / (process_uncertainty + meas_noise)
        return prior_mean + K_t * (measurement - prior_mean)

    @staticmethod
    def _apply_trust_adjustment(candidate: float, trust_delta: float) -> float:
        """Blend system dynamics feedback into the clarity baseline."""

        bounded_delta = max(min(trust_delta, 0.15), -0.15)
        return max(0.0, min(1.0, candidate + bounded_delta))

    def reconcile(
        self, contradictions: Sequence[ContradictionRecord]
    ) -> list[ClarityStatement]:
        """Reconcile a batch of contradictions into clarity statements."""

        statements: list[ClarityStatement] = []
        for record in contradictions:
            insight = self.audit_kernel.assess(record)
            severity_weight = self._severity_weight(record.severity)
            self._drift_tracker.update(insight.confidence, severity_weight)
            current_drift = self._drift_tracker.drift()
            energy_value = getattr(record, "energy", 0.0)
            phase = self.phase_transition.compute_phase(
                self._clarity_baseline, current_drift
            )
            self._current_phase = phase
            measurement_confidence = getattr(record, "confidence", 0.5)
            new_candidate_value = insight.confidence * max(0.1, 1.0 - current_drift)
            updated_clarity = self._kalman_update(
                prior_mean=self._clarity_baseline,
                measurement=new_candidate_value,
                measurement_confidence=measurement_confidence,
            )
            trust_adjustment = self.dynamics.trust_evolution(
                (self._clarity_baseline,),
                0.0,
                insight.confidence,
                energy_value,
                current_drift,
            )[0]
            self._clarity_baseline = self._apply_trust_adjustment(
                updated_clarity, trust_adjustment
            )
            # Compute clarity fuel from contradiction energy
            efficiency = 0.8
            clarity_fuel = efficiency * (
                energy_value if energy_value else self._clarity_baseline * 10.0
            )
            projected_energy = self.dynamics.energy_evolution(
                energy_value, 0.0, lambda _t: energy_value
            )
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
            self._log_drift_metadata(
                record, statement, current_drift, phase, projected_energy
            )
            self._emit_audit_record(record, statement, phase, projected_energy)
        if statements:
            self._emit_diagnostics(statements, len(contradictions))
        return statements

    @staticmethod
    def _severity_weight(severity: str) -> float:
        ladder = {
            "low": 0.6,
            "medium": 0.8,
            "high": 1.0,
            "critical": 1.2,
        }
        return ladder.get(severity.lower(), 0.75)

    def _log_drift_metadata(
        self,
        record: ContradictionRecord,
        statement: ClarityStatement,
        drift: float,
        phase: GovernancePhase,
        projected_energy: float,
    ) -> None:
        if not hasattr(self.ledger, "append_meta"):
            return
        lane = route_to_governance_lane(record)
        payload = {
            "event_type": "METABOLISM_DRIFT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_value": drift,
            "governance_lane": lane,
            "clarity_receipt": statement.to_receipt(),
            "governance_phase": phase.value,
            "projected_energy": projected_energy,
        }
        self.ledger.append_meta("metabolism", payload)

    def _emit_audit_record(
        self,
        record: ContradictionRecord,
        statement: ClarityStatement,
        phase: GovernancePhase,
        projected_energy: float,
    ) -> None:
        payload = {
            "event_type": "METABOLISM_RECONCILIATION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_seed": self._engine_seed,
            "ordered_inputs": [claim.claim_id for claim in record.sorted_pair()],
            "contradiction_delta": record.delta,
            "statement": statement.to_receipt(),
            "governance_phase": phase.value,
            "contradiction_energy": getattr(record, "energy", 0.0),
            "contextual_kappa": getattr(record, "kappa", 0.0),
            "projected_energy": projected_energy,
        }
        self._write_jsonl(self._audit_log_path, payload)

    def _emit_diagnostics(
        self, statements: list[ClarityStatement], total_inputs: int
    ) -> None:
        diagnostics = {
            "event_type": "METABOLISM_DIAGNOSTICS",
            "date": datetime.now(timezone.utc).date().isoformat(),
            "engine_seed": self._engine_seed,
            "processed_events": len(statements),
            "input_count": total_inputs,
            "average_confidence": round(mean(s.confidence for s in statements), 6),
            "clarity_total": round(sum(s.clarity_fuel for s in statements), 6),
            "governance_phase": self._current_phase.value,
        }
        self._write_jsonl(self._diagnostics_path, diagnostics)

    @staticmethod
    def _write_jsonl(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def load_contradictions(path: Path) -> list[ContradictionRecord]:
    """Load contradiction records from a JSON file."""

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):  # pragma: no cover - defensive
        raise ValueError("Expected list of contradiction records")
    return [_record_from_payload(item) for item in data]


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile contradictions into clarity statements"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to a JSON file containing contradiction records",
    )
    parser.add_argument(
        "--export-ledger",
        dest="export_ledger",
        type=Path,
        help="Optional path to write the generated ledger receipts as JSONL",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a built-in reconciliation cycle using sample contradictions",
    )
    return parser


def _emit_ledger(ledger: Ledger, path: Path) -> None:
    ledger.export(path)


def _run_self_test() -> None:
    """Execute a minimal reconciliation cycle to validate dependencies."""

    timestamp = datetime.now(timezone.utc)
    claim_one = Claim(
        claim_id="c-001",
        subject="grid",
        metric="voltage",
        value=118.4,
        unit="V",
        timestamp=timestamp,
        source="sensor_a",
        context={"region": "north"},
    )
    claim_two = Claim(
        claim_id="c-002",
        subject="grid",
        metric="voltage",
        value=123.9,
        unit="V",
        timestamp=timestamp,
        source="sensor_b",
        context={"region": "north"},
    )
    record = ContradictionRecord(
        claim_a=claim_one,
        claim_b=claim_two,
        severity="medium",
        delta=0.08,
        reasoning="Voltage readings diverged across sensors",
    )

    engine = ReconciliationEngine(AuditKernel())
    statements = engine.reconcile([record])
    if not statements:
        raise RuntimeError("Self-test failed to generate clarity statements")

    receipts = engine.ledger.receipts()
    print(
        json.dumps(
            {
                "clarity_statement": statements[0].to_receipt(),
                "ledger_receipts": [receipt.to_json() for receipt in receipts],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_cli()
    args = parser.parse_args(argv)

    if args.self_test:
        _run_self_test()
        return

    if args.input is None:
        parser.error("the following arguments are required: input")

    contradictions = load_contradictions(args.input)
    engine = ReconciliationEngine(AuditKernel())
    statements = engine.reconcile(contradictions)

    for statement in statements:
        print(json.dumps(statement.to_receipt(), indent=2, sort_keys=True))

    if args.export_ledger:
        _emit_ledger(engine.ledger, args.export_ledger)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
