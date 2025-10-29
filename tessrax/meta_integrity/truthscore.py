"""Truth scoring utilities for epistemic meta-audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, MutableMapping, Sequence

TruthVerdict = str


@dataclass(slots=True)
class ClaimEvaluation:
    """Evaluation outcome for a single claim."""

    claim: Mapping[str, object]
    verdict: TruthVerdict
    weight: float
    severity: float


class TruthScore:
    """Aggregate claim verifications into epistemic metrics."""

    _SEVERITY_WEIGHTS = {
        "verified": 0.0,
        "unverified": 0.5,
        "contradicted": 1.0,
    }

    def __init__(
        self,
        verifier: Callable[[Mapping[str, object]], TruthVerdict],
        *,
        accuracy_threshold: float = 0.8,
        severity_threshold: float = 0.15,
    ) -> None:
        self._verifier = verifier
        self.accuracy_threshold = accuracy_threshold
        self.severity_threshold = severity_threshold

    def score(
        self,
        claims: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
        *,
        context: Mapping[str, object] | None = None,
    ) -> MutableMapping[str, object]:
        """Return aggregate epistemic metrics for ``claims``."""

        prepared: List[ClaimEvaluation] = []
        total_weight = 0.0
        verified_weight = 0.0
        severity_weight = 0.0
        for claim in claims:
            certainty = self._coerce_certainty(claim.get("certainty", 1.0))
            weight = max(0.0, min(certainty, 1.0)) or 0.001
            verdict = self._normalise_verdict(self._verifier(claim))
            severity = self._SEVERITY_WEIGHTS.get(verdict, 0.5)
            prepared.append(ClaimEvaluation(claim=claim, verdict=verdict, weight=weight, severity=severity))
            total_weight += weight
            if verdict == "verified":
                verified_weight += weight
            elif verdict == "unverified":
                verified_weight += 0.5 * weight
            severity_weight += severity * weight

        if total_weight == 0:
            return {
                "epistemic_integrity": 0.0,
                "accuracy_score": 0.0,
                "severity_score": 1.0,
                "total_claims": 0,
                "breaches": {"accuracy": True, "severity": True},
                "evaluations": [],
                "context": dict(context or {}),
            }

        accuracy_score = verified_weight / total_weight
        severity_score = severity_weight / total_weight
        epistemic_integrity = max(0.0, min(1.0, accuracy_score * (1 - severity_score)))
        breaches = {
            "accuracy": accuracy_score < self.accuracy_threshold,
            "severity": severity_score > self.severity_threshold,
        }
        report = {
            "epistemic_integrity": epistemic_integrity,
            "accuracy_score": accuracy_score,
            "severity_score": severity_score,
            "total_claims": len(prepared),
            "thresholds": {
                "accuracy": self.accuracy_threshold,
                "severity": self.severity_threshold,
            },
            "breaches": breaches,
            "evaluations": [
                {
                    "claim_id": evaluation.claim.get("claim_id"),
                    "verdict": evaluation.verdict,
                    "weight": evaluation.weight,
                    "severity": evaluation.severity,
                }
                for evaluation in prepared
            ],
            "context": dict(context or {}),
        }
        return report

    @staticmethod
    def _normalise_verdict(verdict: TruthVerdict) -> TruthVerdict:
        if not isinstance(verdict, str):
            return "unverified"
        return verdict.strip().lower() or "unverified"

    @staticmethod
    def _coerce_certainty(raw_value: object) -> float:
        """Best-effort conversion of arbitrary certainty values to ``float``."""

        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value)
            except ValueError:
                return 1.0
        return 1.0


__all__ = ["TruthScore", "TruthVerdict", "ClaimEvaluation"]
