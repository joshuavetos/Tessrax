"""Audit kernel utilities used by Tessrax governance pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["AuditKernel", "AuditInsight", "audit_confidence", "__version__"]

__version__ = "0.1.0"


@dataclass(frozen=True)
class AuditInsight:
    statement: str
    confidence: float


class AuditKernel:
    """Translate metric payloads into structured audit insights."""

    _weights: Mapping[str, float] = {
        "integrity": 0.4,
        "completeness": 0.3,
        "falsifiability": 0.3,
    }

    def assess(self, record: object) -> AuditInsight:
        metrics = self._extract_metrics(record)
        confidence = audit_confidence(metrics)
        statement = self._render_statement(record, metrics, confidence)
        return AuditInsight(statement=statement, confidence=confidence)

    def batch(self, records: Sequence[object]) -> Sequence[AuditInsight]:
        return [self.assess(record) for record in records]

    def _extract_metrics(self, record: object) -> Mapping[str, float]:
        mapping: MutableMapping[str, float] = {}
        if isinstance(record, Mapping):
            candidate: Mapping[str, Any] = record
        else:
            candidate = getattr(record, "metrics", {}) if hasattr(record, "metrics") else {}
            if not isinstance(candidate, Mapping):
                candidate = {
                    key: getattr(record, key)
                    for key in self._weights
                    if hasattr(record, key)
                }
        for key in self._weights:
            value = candidate.get(key)
            if value is None:
                continue
            try:
                mapping[key] = float(value)
            except (TypeError, ValueError):
                continue
        return mapping

    def _render_statement(
        self,
        record: object,
        metrics: Mapping[str, float],
        confidence: float,
    ) -> str:
        metric_clause = ", ".join(f"{name}={value:.2f}" for name, value in sorted(metrics.items()))
        if not metric_clause:
            metric_clause = "no metrics provided"
        return f"Audit assessment for {record!r}: {metric_clause} → confidence {confidence:.3f}"


def audit_confidence(metrics: Mapping[str, float]) -> float:
    """Compute a weighted confidence score from audit metrics."""

    weights = {"integrity": 0.4, "completeness": 0.3, "falsifiability": 0.3}
    numerator = 0.0
    denominator = 0.0
    for key, weight in weights.items():
        if key not in metrics:
            continue
        value = max(0.0, min(1.0, float(metrics[key])))
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return 0.5
    return max(0.0, min(1.0, numerator / denominator))
