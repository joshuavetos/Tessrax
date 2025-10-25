"""Audit kernel heuristics for Tessrax reconciliation flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tessrax.types import ContradictionRecord


@dataclass
class AuditInsights:
    """Structured insights returned by the :class:`AuditKernel`."""

    narrative: str
    confidence: float


class AuditKernel:
    """Derive audit signals that feed the reconciliation engine."""

    _severity_prior = {
        "low": 0.35,
        "medium": 0.55,
        "high": 0.75,
        "critical": 0.9,
    }

    def __init__(self, delta_weight: float = 0.25) -> None:
        self.delta_weight = max(0.0, min(delta_weight, 0.5))

    def assess(self, contradiction: ContradictionRecord) -> AuditInsights:
        """Return a narrative and confidence score for a contradiction."""

        confidence = self._confidence_score(contradiction)
        narrative = self._narrative(contradiction)
        contradiction.confidence = confidence
        return AuditInsights(narrative=narrative, confidence=confidence)

    def _confidence_score(self, contradiction: ContradictionRecord) -> float:
        base = self._severity_prior.get(contradiction.severity.lower(), 0.5)
        delta_term = min(max(abs(contradiction.delta), 0.0), 1.0)
        confidence = base + self.delta_weight * delta_term
        return max(0.0, min(confidence, 1.0))

    def _narrative(self, contradiction: ContradictionRecord) -> str:
        claim_ids = ", ".join(claim.claim_id for claim in contradiction.sorted_pair())
        return (
            "Audit Kernel synthesized a clarity path for claims "
            f"{claim_ids} (severity={contradiction.severity}). "
            f"Reasoning: {contradiction.reasoning}"
        )


def batch_assess(kernel: AuditKernel, contradictions: Sequence[ContradictionRecord]) -> Sequence[AuditInsights]:
    """Evaluate a batch of contradictions and yield audit insights."""

    return [kernel.assess(item) for item in contradictions]
