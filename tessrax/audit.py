"""Audit kernel heuristics for Tessrax reconciliation flows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tessrax.core import audit_kernel

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
        metrics = {
            "integrity": max(0.0, min(1.0, 1.0 - min(abs(contradiction.delta), 1.0))),
            "completeness": self._completeness_metric(contradiction),
            "falsifiability": self._falsifiability_metric(contradiction),
        }
        kernel_score = audit_kernel.audit_confidence(metrics)
        base = self._severity_prior.get(contradiction.severity.lower(), 0.5)
        delta_term = min(max(abs(contradiction.delta), 0.0), 1.0)
        severity_component = max(0.0, min(base + self.delta_weight * delta_term, 1.0))
        confidence = (0.6 * kernel_score) + (0.4 * severity_component)
        return max(0.0, min(confidence, 1.0))

    @staticmethod
    def _completeness_metric(contradiction: ContradictionRecord) -> float:
        reasoning = contradiction.reasoning or ""
        if not reasoning.strip():
            return 0.6
        richness = min(len(reasoning.strip()) / 280.0, 1.0)
        return max(0.6, min(1.0, 0.6 + 0.4 * richness))

    @staticmethod
    def _falsifiability_metric(contradiction: ContradictionRecord) -> float:
        energy = getattr(contradiction, "energy", 0.0) or 0.0
        kappa = getattr(contradiction, "kappa", 0.0) or 0.0
        energy_term = energy / (abs(energy) + 1.0)
        kappa_term = min(abs(kappa), 1.0)
        base = 0.55 + 0.3 * energy_term
        return max(base, min(1.0, 0.55 + 0.35 * kappa_term))

    def _narrative(self, contradiction: ContradictionRecord) -> str:
        claim_ids = ", ".join(claim.claim_id for claim in contradiction.sorted_pair())
        return (
            "Audit Kernel synthesized a clarity path for claims "
            f"{claim_ids} (severity={contradiction.severity}). "
            f"Reasoning: {contradiction.reasoning}"
        )


def batch_assess(
    kernel: AuditKernel, contradictions: Sequence[ContradictionRecord]
) -> Sequence[AuditInsights]:
    """Evaluate a batch of contradictions and yield audit insights."""

    return [kernel.assess(item) for item in contradictions]
