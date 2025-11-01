"""Validation routines for Truth API requests."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException, status

from tessrax_truth_api.engine.contradiction_engine import ContradictionEngine


@dataclass
class ValidationResult:
    score: float
    verdict: str
    status: str


class ValidationService:
    """Coordinates claim validation using the contradiction engine."""

    def __init__(self, engine: ContradictionEngine, thresholds: tuple[float, float]):
        self._engine = engine
        self._thresholds = thresholds

    def validate_claim_pair(self, claim_a: str, claim_b: str) -> ValidationResult:
        if not claim_a or not claim_b:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Claims must be provided",
            )

        score = self._engine.score(claim_a, claim_b)
        verdict = self._engine.verdict(score, self._thresholds)
        status_value = "verified" if verdict != "unknown" else "unknown"
        return ValidationResult(score=score, verdict=verdict, status=status_value)
