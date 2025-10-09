"""
Performance Regression Test — ensures CE-MOD-66 scales within limits.
"""
import time
from ce_mod_66 import detect_contradictions, score_stability

BASELINE = {
    1000: 0.5,   # seconds
    5000: 2.0,
    10000: 5.0
}

def generate_claims(n):
    return [{"agent": f"A{i}", "claim": f"claim{i%2}", "type": "normative"} for i in range(n)]

def benchmark_step(n):
    claims = generate_claims(n)
    t0 = time.time()
    G = detect_contradictions(claims)
    _ = score_stability(G)
    return time.time() - t0

def test_performance_regression():
    for n, baseline_time in BASELINE.items():
        elapsed = benchmark_step(n)
        assert elapsed < baseline_time * 1.25, f"Regression: {n} agents took {elapsed:.2f}s > {baseline_time*1.25:.2f}s"