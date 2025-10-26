"""Embedding inspired contradiction scoring utilities."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Tuple


class ContradictionEngine:
    """Lightweight heuristic engine that simulates contradiction scoring."""

    def __init__(self, *, contradiction_bias: float = 0.55) -> None:
        self.contradiction_bias = contradiction_bias

    def _normalise(self, value: str) -> str:
        return " ".join(value.lower().strip().split())

    def _negated_pair(self, a: str, b: str) -> bool:
        negations = {"not", "never", "no", "cannot", "can't"}
        tokens_a = set(self._normalise(a).split())
        tokens_b = set(self._normalise(b).split())
        return bool(tokens_a & negations) ^ bool(tokens_b & negations)

    def score(self, claim_a: str, claim_b: str) -> float:
        """Return a score between 0 and 1."""

        normal_a, normal_b = self._normalise(claim_a), self._normalise(claim_b)
        if not normal_a or not normal_b:
            return 0.0

        if normal_a == normal_b:
            return 0.0

        matcher = SequenceMatcher(None, normal_a, normal_b)
        distance = 1.0 - matcher.quick_ratio()
        if self._negated_pair(normal_a, normal_b):
            distance = min(1.0, distance + self.contradiction_bias)
        return round(distance, 4)

    def verdict(self, score: float, thresholds: Tuple[float, float]) -> str:
        """Return a categorical verdict based on thresholds."""

        contradiction_threshold, unknown_threshold = thresholds
        if score >= contradiction_threshold:
            return "contradiction"
        if score <= unknown_threshold:
            return "aligned"
        return "unknown"
