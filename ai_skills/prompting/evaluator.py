"""Scoring helpers for comparing model guesses against reference outputs."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class EvaluationResult:
    """Structured result returned by :class:`Evaluator`."""

    match: bool
    similarity: float
    explanation: str


class Evaluator:
    """Compute lightweight, deterministic similarity scores."""

    def score(self, guess: str, truth: str) -> EvaluationResult:
        """Compare ``guess`` to ``truth`` and return a deterministic score."""

        guess_normalised = guess.strip().lower()
        truth_normalised = truth.strip().lower()
        similarity = SequenceMatcher(None, guess_normalised, truth_normalised).ratio()
        match = guess_normalised == truth_normalised
        explanation = self._build_explanation(match, similarity)
        return EvaluationResult(
            match=match, similarity=similarity, explanation=explanation
        )

    def score_as_dict(self, guess: str, truth: str) -> dict[str, str | float | bool]:
        """Return the evaluation as a plain ``dict`` suitable for JSON serialisation."""

        result = self.score(guess, truth)
        return {
            "match": result.match,
            "similarity": result.similarity,
            "explanation": result.explanation,
        }

    @staticmethod
    def _build_explanation(match: bool, similarity: float) -> str:
        """Return an actionable explanation for the score."""

        if match:
            return "Guesses match exactly after normalisation."
        if similarity == 0:
            return "Guesses do not overlap; review the generated output."
        if similarity < 0.5:
            return "Guesses share limited overlap. Investigate prompt fidelity or revise the truth set."
        return (
            "Guesses are close but not identical. Inspect differences and adjust the prompt or"
            " evaluation criteria."
        )
