"""Ethical drift monitor aligned with Tessrax governance protocols.

All computations follow AEP-001, RVC-001, and POST-AUDIT-001. The
module estimates cosine similarity between historical and proposed
rule corpora to identify ethical shifts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import sqrt

from tessrax.core.governance.receipts import write_receipt


def _tokenize(rule_set: Sequence[str]) -> Counter:
    """Tokenize rule strings deterministically using whitespace splitting."""

    tokens: Counter = Counter()
    for rule in rule_set:
        for token in rule.lower().split():
            tokens[token] += 1
    return tokens


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Compute cosine similarity between two sparse counters."""

    intersection = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in intersection)
    norm_a = sqrt(sum(value * value for value in a.values())) or 1.0
    norm_b = sqrt(sum(value * value for value in b.values())) or 1.0
    return round(numerator / (norm_a * norm_b), 6)


def assess_value_drift(
    previous_rules: Sequence[str], proposed_rules: Sequence[str]
) -> dict[str, float | str]:
    """Assess semantic drift and flag ethical shift events."""

    base = _tokenize(previous_rules)
    candidate = _tokenize(proposed_rules)
    similarity = _cosine_similarity(base, candidate)
    drift = round(1.0 - similarity, 6)
    status = "ethical shift" if drift > 0.15 else "stable"
    metrics = {
        "similarity": similarity,
        "drift": drift,
        "status": status,
    }
    integrity = 0.98 if status != "ethical shift" or drift < 0.5 else 0.9
    write_receipt(
        "tessrax.core.ethics.value_drift_monitor", "verified", metrics, integrity
    )
    return metrics


def _self_test() -> bool:
    """Validate drift detection on simulated rule updates."""

    historical = ["Rules shall remain transparent", "Audit logs immutable"]
    revised = [
        "Rules shall remain transparent",
        "Audit logs immutable",
        "Introduce adaptive oversight",
    ]
    metrics = assess_value_drift(historical, revised)
    assert metrics["drift"] > 0, "Expected drift"  # type: ignore[index]
    assert metrics["status"] == "ethical shift", "Shift should be detected"  # type: ignore[index]
    write_receipt(
        "tessrax.core.ethics.value_drift_monitor.self_test",
        "verified",
        {"drift": metrics["drift"], "status": metrics["status"]},
        0.95,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
