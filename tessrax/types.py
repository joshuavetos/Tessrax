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
        if self.sub_merkle_root is not None:
            payload["sub_merkle_root"] = self.sub_merkle_root
        if self.signature is not None:
            payload["signature"] = self.signature
        return payload
