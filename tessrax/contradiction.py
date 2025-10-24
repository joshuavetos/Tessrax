"""Contradiction detection heuristics."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Sequence

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

    def detect(self, claims: Sequence[Claim]) -> List[ContradictionRecord]:
        grouped: Dict[str, List[Claim]] = defaultdict(list)
        for claim in claims:
            grouped[claim.key()].append(claim)

        contradictions: List[ContradictionRecord] = []
        for key, key_claims in grouped.items():
            if len(key_claims) < 2:
                continue
            for claim_a, claim_b in combinations(sorted(key_claims, key=lambda c: c.claim_id), 2):
                delta = abs(claim_a.value - claim_b.value)
                baseline = max(abs(claim_a.value), abs(claim_b.value), 1.0)
                relative_gap = delta / baseline
                if relative_gap <= self.tolerance:
                    continue
                severity = self._grade(relative_gap)
                reasoning = (
                    f"{claim_a.subject} {claim_a.metric} differs from {claim_b.source} by "
                    f"{relative_gap:.1%} ({claim_a.value} vs {claim_b.value})."
                )
                contradictions.append(
                    ContradictionRecord(
                        claim_a=claim_a,
                        claim_b=claim_b,
                        severity=severity,
                        delta=relative_gap,
                        reasoning=reasoning,
                    )
                )
        return contradictions

    @staticmethod
    def _grade(relative_gap: float) -> str:
        for level, threshold in sorted(SeverityThresholds.items(), key=lambda item: item[1]):
            if relative_gap <= threshold:
                return level
        return "critical"
