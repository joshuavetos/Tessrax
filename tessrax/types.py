"""Typed primitives shared across Tessrax modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

from uuid import UUID

try:  # pragma: no cover - python < 3.12 compatibility
    from uuid import uuid7 as _uuid7_native  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for environments without uuid7
    from uuid import uuid4 as _uuid4

    def _generate_uuid7() -> UUID:
        return _uuid4()
else:

    def _generate_uuid7() -> UUID:
        return _uuid7_native()

Severity = str
Action = str


@dataclass(slots=True)
class Claim:
    """Structured representation of an input statement."""

    claim_id: str
    subject: str
    metric: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    context: Dict[str, str] = field(default_factory=dict)

    def key(self) -> str:
        """Return a composite key for grouping claims."""

        return f"{self.subject}:{self.metric}:{self.unit}".lower()

    def to_json(self) -> Dict[str, object]:
        """Serialise the claim into a JSON-compatible mapping."""

        payload: Dict[str, object] = {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }
        if self.context:
            payload["context"] = dict(self.context)
        return payload


@dataclass(slots=True)
class ContradictionRecord:
    """Conflict detected between two claims."""

    claim_a: Claim
    claim_b: Claim
    severity: Severity
    delta: float
    reasoning: str
    confidence: float = 0.5
    energy: float = 0.0
    kappa: float = 0.0
    contradiction_type: Optional[str] = None

    @property
    def claims(self) -> tuple[Claim, Claim]:
        """Return the ordered claim pair to support semantic classifiers."""

        return (self.claim_a, self.claim_b)

    def sorted_pair(self) -> Iterable[Claim]:
        """Yield claims in deterministic order for reproducibility."""

        return tuple(sorted((self.claim_a, self.claim_b), key=lambda c: c.claim_id))

    def to_json(self) -> Dict[str, object]:
        """Serialise the contradiction and its claims."""

        payload: Dict[str, object] = {
            "claim_a": self.claim_a.to_json(),
            "claim_b": self.claim_b.to_json(),
            "severity": self.severity,
            "delta": self.delta,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "energy": self.energy,
            "kappa": self.kappa,
        }
        if self.contradiction_type is not None:
            payload["contradiction_type"] = self.contradiction_type
        return payload


@dataclass(slots=True)
class GovernanceDecision:
    """Governance kernel decision for a contradiction."""

    contradiction: ContradictionRecord
    action: Action
    clarity_fuel: float
    rationale: str
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    protocol: str = "governance"
    decision_id: UUID = field(default_factory=_generate_uuid7)
    timestamp_token: Optional[str] = None
    signature: Optional[str] = None

    def to_summary(self) -> Dict[str, object]:
        """Short JSON-serialisable representation."""

        return {
            "event_type": "CONTRADICTION_DECISION",
            "timestamp": self.issued_at.isoformat(),
            "action": self.action,
            "severity": self.contradiction.severity,
            "clarity_fuel": round(self.clarity_fuel, 3),
            "subject": self.contradiction.claim_a.subject,
            "metric": self.contradiction.claim_a.metric,
            "rationale": self.rationale,
            "decision_id": str(self.decision_id),
            "protocol": self.protocol,
            **({"timestamp_token": self.timestamp_token} if self.timestamp_token else {}),
            **({"signature": self.signature} if self.signature else {}),
        }

    def canonical_document(self) -> Dict[str, object]:
        """Return the canonical payload used for signing and timestamping."""

        payload = self.to_summary().copy()
        payload.pop("signature", None)
        payload.pop("timestamp_token", None)
        return payload

    def to_decision_payload(self) -> Dict[str, object]:
        """Return a detailed representation including contradiction context."""

        payload: Dict[str, object] = {
            "action": self.action,
            "clarity_fuel": round(self.clarity_fuel, 3),
            "protocol": self.protocol,
            "issued_at": self.issued_at.isoformat(),
            "contradiction": self.contradiction.to_json(),
            "severity": self.contradiction.severity,
            "decision_id": str(self.decision_id),
        }
        if self.timestamp_token is not None:
            payload["timestamp_token"] = self.timestamp_token
        if self.signature is not None:
            payload["signature"] = self.signature
        return payload


@dataclass(slots=True)
class LedgerReceipt:
    """Append-only ledger entry for governance actions."""

    decision: GovernanceDecision
    prev_hash: str
    hash: str
    signature: Optional[str]
    sub_merkle_root: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        payload = self.decision.to_summary()
        payload.update({
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        })
        decision_payload = self.decision.to_decision_payload()
        payload["decision"] = decision_payload
        contradiction_payload = decision_payload.get("contradiction")
        if contradiction_payload is not None:
            payload.setdefault("contradiction", contradiction_payload)
        if self.sub_merkle_root is not None:
            payload["sub_merkle_root"] = self.sub_merkle_root
        if self.signature is not None:
            payload["signature"] = self.signature
        return payload
