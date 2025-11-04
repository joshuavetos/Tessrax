"""Golden-set verification tests for epistemic metric helpers."""

import math
import pytest

from core.epistemic_metrics import calculate_entropy, compute_legitimacy


def test_entropy_known_values():
    probs = [0.5, 0.5]
    assert math.isclose(calculate_entropy(probs), 1.0, rel_tol=1e-9)


def test_entropy_validation_sum_error():
    with pytest.raises(ValueError):
        calculate_entropy([0.4, 0.4])


def test_legitimacy_known_values():
    result = compute_legitimacy(1.0, 0.5, 0.5, 0.5)
    expected = 0.4 * 1.0 + 0.2 * 0.5 + 0.2 * 0.5 + 0.2 * 0.5
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_legitimacy_bounds_check():
    with pytest.raises(ValueError):
        compute_legitimacy(1.2, 0.5, 0.5, 0.5)
