"""Epistemic health metrics with formally documented provenance.

This module provides canonical implementations for the Tessrax epistemic metrics
that are audited as part of the v16 hardening programme.  Each function includes
the closed-form definition that is also recorded in :mod:`docs.metrics_specification`
and validated against the JSON schema test vectors.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import List, Tuple


def compute_integrity(outcomes: List[float], sigma_max: float | None = None) -> float:
    """Return the bounded integrity score.

    The integrity metric is defined as ``1 - (σ_t / σ_max)`` where ``σ_t`` is the
    population standard deviation of the supplied outcomes and ``σ_max`` is the
    admissible maximum spread (defaults to half of the min/max range).  The result
    is clamped to the :math:`[0, 1]` interval to guarantee reproducibility.
    """
    if len(outcomes) < 2:
        return 1.0
    sigma_t = statistics.pstdev(outcomes)
    if sigma_max is None:
        spread = max(outcomes) - min(outcomes)
        sigma_max = spread / 2 if spread > 0 else 1e-9
    raw = 1.0 - (sigma_t / sigma_max if sigma_max > 0 else 0.0)
    return max(0.0, min(1.0, raw))


def compute_drift(history: List[Tuple[float, float]]) -> float:
    """Return the absolute drift between the latest value and the prior mean."""
    if len(history) < 2:
        return 0.0
    vals = [v for _, v in history]
    latest = vals[-1]
    mean_prior = statistics.fmean(vals[:-1])
    drift = abs(latest - mean_prior)
    return max(0.0, min(1.0, drift))


def compute_severity(expected: List[float], observed: List[float]) -> float:
    """Return the bounded mean absolute error across expected and observed values."""
    if not expected or not observed or len(expected) != len(observed):
        return 0.0
    diffs = [abs(a - b) for a, b in zip(expected, observed)]
    mae = statistics.fmean(diffs)
    return max(0.0, min(1.0, mae))


def compute_entropy(labels: list[str]) -> float:
    """Return the normalised Shannon entropy of the supplied labels."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    H = -sum((n / total) * math.log2(n / total) for n in counts.values())
    max_H = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return max(0.0, min(1.0, H / max_H))
