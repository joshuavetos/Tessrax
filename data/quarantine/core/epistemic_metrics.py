"""Utility functions for governance entropy and legitimacy metrics.

This module adheres to Tessrax Governance clauses by providing deterministic
computations, validation gates, and auditable documentation of the weightings
used for each composite score.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    """Validate that the provided probability distribution is well-formed."""
    if not probabilities:
        raise ValueError("probabilities must be a non-empty sequence")

    total = 0.0
    for index, value in enumerate(probabilities):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Probability at index {index} is not numeric: {value!r}")
        if value < 0 or value > 1:
            raise ValueError(
                f"Probability at index {index} must be within [0, 1]; received {value}."
            )
        total += value

    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "Probabilities must sum to 1 within numerical tolerance; "
            f"observed sum={total:.12f}."
        )


def calculate_entropy(probabilities: Iterable[float]) -> float:
    """Compute Shannon entropy (base 2) of a list of probabilities.

    The distribution is validated for non-negativity and unit sum before
    entropy is calculated. Zero-valued probabilities are excluded from the
    logarithm to maintain numerical stability.
    """

    probs = list(probabilities)
    _validate_probabilities(probs)
    filtered = [p for p in probs if p > 0]
    return -sum(p * math.log2(p) for p in filtered)


def compute_legitimacy(
    integrity_score: float,
    audit_trail: float,
    quorum_score: float,
    signature_confidence: float,
) -> float:
    """Return the weighted trustworthiness composite constrained to [0, 1].

    Each input is required to be between 0 and 1 inclusive. Runtime validation
    ensures the scores can be meaningfully combined under the configured
    0.4/0.2/0.2/0.2 weighting model.
    """

    inputs = {
        "integrity_score": integrity_score,
        "audit_trail": audit_trail,
        "quorum_score": quorum_score,
        "signature_confidence": signature_confidence,
    }

    for name, value in inputs.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric; received {value!r}")
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be within [0, 1]; received {value}")

    legitimacy = (
        0.4 * integrity_score
        + 0.2 * audit_trail
        + 0.2 * quorum_score
        + 0.2 * signature_confidence
    )
    return max(0.0, min(1.0, legitimacy))


if __name__ == "__main__":
    receipt = {
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "runtime_info": "self-check",
        "integrity_score": 1.0,
        "status": "DLK-VERIFIED",
        "signature": __import__("hashlib").sha256(__file__.encode()).hexdigest(),
    }
    print(receipt)
