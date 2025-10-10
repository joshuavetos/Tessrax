"""
Benchmark Runner
Merged benchmark_runner.py + test_performance_regression.py snippets.
Runs performance benchmarks for Tessrax modules.
"""

import time
import psutil
import os
from tessrax.core.engine_core import detect_contradictions, score_stability


def measure_memory_mb():
    """Return current process memory (MB)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def benchmark(size: int = 1000):
    """Generate synthetic dataset and measure time + memory usage."""
    agents = [{"agent": f"A{i}", "claim": "X" if i % 2 == 0 else "Y"} for i in range(size)]
    start_time = time.perf_counter()
    mem_before = measure_memory_mb()
    G = detect_contradictions(agents)
    _ = score_stability(G)
    mem_after = measure_memory_mb()
    elapsed = time.perf_counter() - start_time
    return {"agents": size, "time_s": elapsed, "mem_mb": mem_after - mem_before}


if __name__ == "__main__":
    for n in [1000, 5000, 10000]:
        result = benchmark(n)
        print(f"{n} agents → {result['time_s']:.3f}s, {result['mem_mb']:.1f}MB")
