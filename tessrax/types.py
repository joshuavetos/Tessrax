"""Typed primitives shared across Tessrax modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

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
        }


@dataclass(slots=True)
class LedgerReceipt:
    """Append-only ledger entry for governance actions."""

    decision: GovernanceDecision
    prev_hash: str
    hash: str
    signature: Optional[str]

    def to_json(self) -> Dict[str, object]:
        payload = self.decision.to_summary()
        payload.update({
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "signature": self.signature or "pending",
        })
        return payload
