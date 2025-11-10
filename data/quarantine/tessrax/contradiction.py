"""Contradiction detection heuristics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import combinations

from tessrax.physics.context import AgentAlignment, ContextualStiffness
from tessrax.physics.energy import ContradictionEnergy
from tessrax.types import Claim, ContradictionRecord

SeverityThresholds = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.3,
}


class ContradictionEngine:
    """Detect contradictory claims using simple numeric heuristics."""

    def __init__(self, tolerance: float = 0.05) -> None:
        self.tolerance = tolerance
        self.energy_model = ContradictionEnergy()
        self.stiffness = ContextualStiffness()
        self.alignment = AgentAlignment()

    def detect(self, claims: Sequence[Claim]) -> list[ContradictionRecord]:
        grouped: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            grouped[claim.key()].append(claim)

        contradictions: list[ContradictionRecord] = []
        for key, key_claims in grouped.items():
            if len(key_claims) < 2:
                continue
            for claim_a, claim_b in combinations(
                sorted(key_claims, key=lambda c: c.claim_id), 2
            ):
                delta = abs(claim_a.value - claim_b.value)
                baseline = max(abs(claim_a.value), abs(claim_b.value), 1.0)
                relative_gap = delta / baseline
                alignment_score = self.alignment.compute(claim_a.source, claim_b.source)
                self.alignment.update(
                    claim_a.source, claim_b.source, relative_gap <= self.tolerance
                )
                if relative_gap <= self.tolerance:
                    continue
                severity = self._grade(relative_gap)
                reasoning = (
                    f"{claim_a.subject} {claim_a.metric} differs from {claim_b.source} by "
                    f"{relative_gap:.1%} ({claim_a.value} vs {claim_b.value})."
                )
                temporal_distance = self._temporal_distance_days(
                    claim_a.timestamp, claim_b.timestamp
                )
                contextual_stiffness = self.stiffness.compute(
                    claim_a.subject,
                    claim_a.metric,
                    temporal_distance,
                )
                base_probability = self._base_probability(severity)
                kappa = self.energy_model.compute_kappa(
                    contextual_stiffness, alignment_score, base_probability
                )
                energy = self.energy_model.contradiction_energy(relative_gap, kappa)
                contradictions.append(
                    ContradictionRecord(
                        claim_a=claim_a,
                        claim_b=claim_b,
                        severity=severity,
                        delta=relative_gap,
                        reasoning=reasoning,
                        energy=energy,
                        kappa=kappa,
                    )
                )
        return contradictions

    @staticmethod
    def _grade(relative_gap: float) -> str:
        for level, threshold in sorted(
            SeverityThresholds.items(), key=lambda item: item[1]
        ):
            if relative_gap <= threshold:
                return level
        return "critical"

    @staticmethod
    def _temporal_distance_days(first: datetime, second: datetime) -> float:
        """Return absolute temporal distance between claims in days."""

        delta = first - second
        return abs(delta.total_seconds()) / 86_400.0

    @staticmethod
    def _base_probability(severity: str) -> float:
        """Map severity labels to a prior contradiction probability."""

        priors = {
            "low": 0.35,
            "medium": 0.55,
            "high": 0.75,
            "critical": 0.9,
        }
        return priors.get(severity.lower(), 0.5)
