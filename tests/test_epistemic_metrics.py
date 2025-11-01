from tessrax.metrics.epistemic_health import (
    compute_drift,
    compute_entropy,
    compute_integrity,
    compute_severity,
)


def test_integrity_range():
    vals = [0.9, 0.92, 0.88, 0.91]
    I = compute_integrity(vals)
    assert 0 <= I <= 1


def test_drift_behavior():
    history = [(0, 0.8), (1, 0.85), (2, 0.83)]
    d = compute_drift(history)
    assert 0 <= d <= 1


def test_severity_monotonic():
    e = [0.9, 0.8, 0.7]
    o1 = [0.9, 0.8, 0.7]
    o2 = [0.5, 0.5, 0.5]
    s1 = compute_severity(e, o1)
    s2 = compute_severity(e, o2)
    assert s1 <= s2


def test_entropy_normalized():
    labels = ["semantic", "semantic", "procedural", "federated"]
    H = compute_entropy(labels)
    assert 0 <= H <= 1
