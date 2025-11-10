"""Governance façade providing a ``resolve`` helper for runtime tests.

The façade preserves the public ``GovernanceKernel`` surface while exposing a
``resolve`` helper that consumes contradiction payloads produced by
``core.contradiction_engine.detect_contradictions``.  The helper executes the
canonical Tessrax governance flow, records the decision for ledger persistence,
and emits DLK-verified summaries compliant with the Codex Iron-Law directives.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import log2
from typing import Mapping, Sequence

from tessrax.contradiction import Claim
from tessrax.governance import (
    ContradictionRecord,
    GovernanceDecision,
    GovernanceKernel,
    classify_contradiction,
    route_to_governance_lane,
)

from .contradiction_engine import last_contradictions

__all__ = [
    "GovernanceKernel",
    "classify_contradiction",
    "route_to_governance_lane",
    "resolve",
    "last_decision",
]


@dataclass(slots=True)
class _DecisionEnvelope:
    """Capture both the decision object and its serialised summary."""

    decision: GovernanceDecision
    summary: dict[str, object]


_last_decision_envelope: _DecisionEnvelope | None = None


def _iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _rehydrate_claim(payload: Mapping[str, object]) -> Claim:
    if isinstance(payload, Claim):
        return payload
    required = {"claim_id", "subject", "metric", "value", "unit", "timestamp", "source"}
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ValueError(f"Contradiction claim missing fields: {', '.join(missing)}")
    timestamp = payload["timestamp"]
    if isinstance(timestamp, str):
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp_dt = parsed.astimezone(timezone.utc)
    elif isinstance(timestamp, datetime):
        timestamp_dt = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    else:
        raise ValueError("Contradiction claim has non-serialisable timestamp")
    context_raw = payload.get("context", {})
    context = dict(context_raw) if isinstance(context_raw, Mapping) else {}
    return Claim(
        claim_id=str(payload["claim_id"]),
        subject=str(payload["subject"]),
        metric=str(payload["metric"]),
        value=float(payload["value"]),
        unit=str(payload["unit"]),
        timestamp=timestamp_dt,
        source=str(payload["source"]),
        context=context,
    )


def _rehydrate_contradiction(payload: Mapping[str, object]) -> ContradictionRecord:
    if isinstance(payload, ContradictionRecord):
        return payload
    if "claim_a" not in payload or "claim_b" not in payload:
        raise ValueError("Contradiction payload must include claim_a and claim_b")
    claim_a = _rehydrate_claim(payload["claim_a"])
    claim_b = _rehydrate_claim(payload["claim_b"])
    severity = str(payload.get("severity", "medium"))
    delta = float(payload.get("delta", 0.0))
    reasoning = str(payload.get("reasoning", "no reasoning provided"))
    confidence = float(payload.get("confidence", 0.5))
    energy = float(payload.get("energy", 0.0))
    kappa = float(payload.get("kappa", 0.0))
    contradiction_type = payload.get("contradiction_type")
    if contradiction_type is not None:
        contradiction_type = str(contradiction_type)
    return ContradictionRecord(
        claim_a=claim_a,
        claim_b=claim_b,
        severity=severity,
        delta=delta,
        reasoning=reasoning,
        confidence=confidence,
        energy=energy,
        kappa=kappa,
        contradiction_type=contradiction_type,
    )


def _entropy_from_delta(delta: float) -> float:
    clamped = max(min(delta, 0.999999), 1e-6)
    return round(-clamped * log2(clamped) - (1 - clamped) * log2(1 - clamped), 6)


def _build_summary(decision: GovernanceDecision) -> dict[str, object]:
    contradiction = decision.contradiction
    entropy = _entropy_from_delta(contradiction.delta)
    summary = {
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
        "status": "DLK-VERIFIED",
        "verdict": decision.action,
        "severity": contradiction.severity,
        "entropy": entropy,
        "clarity_fuel": round(decision.clarity_fuel, 6),
        "timestamp": _iso_timestamp(decision.issued_at),
        "decision_id": str(decision.decision_id),
        "subject": contradiction.claim_a.subject,
        "metric": contradiction.claim_a.metric,
        "reasoning": decision.rationale,
        "runtime_info": {
            "delta": round(contradiction.delta, 6),
            "energy": round(contradiction.energy, 6),
            "kappa": round(contradiction.kappa, 6),
        },
    }
    return summary


def resolve(
    contradictions: Sequence[Mapping[str, object] | ContradictionRecord],
    *,
    kernel: GovernanceKernel | None = None,
) -> dict[str, object]:
    """Resolve contradictions via the Tessrax governance kernel.

    The function performs runtime validation, resolves the most severe
    contradiction, and records the resulting decision so the ledger façade can
    persist it.  Downstream consumers receive a DLK-verified JSON payload.
    """

    if not contradictions:
        stored = last_contradictions()
        if not stored:
            raise ValueError("No contradictions supplied for governance resolution")
        contradictions = stored

    records: list[ContradictionRecord] = []
    for entry in contradictions:
        if isinstance(entry, ContradictionRecord):
            records.append(entry)
        elif isinstance(entry, Mapping):
            records.append(_rehydrate_contradiction(entry))
        else:
            raise TypeError(
                "Governance resolution expects mappings or ContradictionRecord instances"
            )

    if not records:
        raise ValueError("Unable to resolve governance without contradiction records")

    target = max(records, key=lambda record: record.delta)
    active_kernel = kernel or GovernanceKernel()
    decision = active_kernel.process(target)
    summary = _build_summary(decision)

    global _last_decision_envelope
    _last_decision_envelope = _DecisionEnvelope(decision=decision, summary=summary)
    return dict(summary)


def last_decision() -> GovernanceDecision | None:
    return _last_decision_envelope.decision if _last_decision_envelope else None


