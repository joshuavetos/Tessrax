"""Analytics helpers for epistemic integrity telemetry."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping


def compute_epistemic_metrics(reports: Iterable[Mapping[str, float]]) -> MutableMapping[str, float]:
    """Compute aggregate drift and resilience metrics from audit ``reports``."""

    reports = list(reports)
    if not reports:
        raise ValueError("compute_epistemic_metrics requires at least one report")

    first = reports[0]
    last = reports[-1]
    drift = abs(float(last.get("epistemic_integrity", 0.0)) - float(first.get("epistemic_integrity", 0.0)))
    resilience_hits = 0
    hallucination_hits = 0
    for report in reports:
        accuracy = float(report.get("accuracy_score", 0.0))
        severity = float(report.get("severity_score", 0.0))
        if accuracy > 0.8:
            resilience_hits += 1
        if severity > 0.5:
            hallucination_hits += 1
    total = len(reports)
    resilience = resilience_hits / total
    hallucination_rate = hallucination_hits / total
    return {
        "epistemic_drift": round(drift, 3),
        "adversarial_resilience": round(resilience, 3),
        "hallucination_rate": round(hallucination_rate, 3),
    }


__all__ = ["compute_epistemic_metrics"]
