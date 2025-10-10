"""
Tessrax Core Test Suite (TEST-MOD-04)
-------------------------------------

Unified diagnostic and benchmark harness for the Tessrax Core stack.
Validates end-to-end performance across modules:

    • CE-MOD-66  — Contradiction Engine
    • GK-MOD-01  — Governance Kernel
    • VIS-MOD-03 — Visualization Scaffold (optional)

Runs both functional tests and benchmark simulations with synthetic data.

Version: TEST-MOD-04-R2
Author: Tessrax LLC
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict

import pytest
import networkx as nx
import psutil

from tessrax.core import contradiction_engine as ce
from tessrax.core import governance_kernel as gk


# === UTILITY =================================================================

def _print_banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def _measure_memory() -> float:
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


# === FUNCTIONAL TESTS ========================================================

def test_contradiction_detection():
    """Ensure contradictions are correctly identified."""
    claims = [
        {"agent": "GPT", "claim": "AI must explain its actions.", "type": "normative"},
        {"agent": "Claude", "claim": "AI must not explain its actions.", "type": "normative"},
    ]
    G = ce.detect_contradictions(claims)
    assert G.number_of_edges() == 1, "Should detect one contradiction edge"


def test_stability_calculation():
    """Verify stability index scaling."""
    claims = [
        {"agent": "A", "claim": "X", "type": "normative"},
        {"agent": "B", "claim": "Y", "type": "normative"},
    ]
    G = ce.detect_contradictions(claims)
    stability = ce.score_stability(G)
    assert 0 <= stability <= 1, "Stability must be within [0,1]"


def test_governance_routing():
    """Ensure governance lanes are correctly classified."""
    claims = [
        {"agent": "A", "claim": "X", "type": "normative"},
        {"agent": "B", "claim": "Y", "type": "normative"},
        {"agent": "C", "claim": "Y", "type": "normative"},
    ]
    G = ce.detect_contradictions(claims)
    stability = ce.score_stability(G)
    event = gk.route(G, stability)
    assert event.governance_lane in [lane.value for lane in gk.GovernanceLane], "Invalid governance lane"


# === FAILURE MODE TESTS ======================================================

def test_ledger_chain_integrity(tmp_path):
    """Test hash chain integrity in the ledger."""
    ledger_path = tmp_path / "ledger.jsonl"
    claims = [
        {"agent": "A", "claim": "X", "type": "normative"},
        {"agent": "B", "claim": "Y", "type": "normative"},
    ]
    G = ce.detect_contradictions(claims)
    stability = ce.score_stability(G)
    ce.log_to_ledger(G, stability, str(ledger_path))

    with open(ledger_path, "r") as f:
        first = json.loads(f.readline())

    ce.log_to_ledger(G, stability, str(ledger_path))

    with open(ledger_path, "r") as f:
        lines = [json.loads(line) for line in f]
    assert lines[-1]["prev_hash"] == first["hash"], "Hash chain must be intact"


# === PERFORMANCE BENCHMARKS ==================================================

def benchmark_contradiction_engine(num_agents: int = 1000) -> Dict[str, float]:
    """
    Benchmark CE-MOD-66 on synthetic datasets.
    """
    claims = [{"agent": f"A{i}", "claim": "X" if i % 2 == 0 else "Y", "type": "normative"} for i in range(num_agents)]
    start_time = time.time()
    mem_before = _measure_memory()

    G = ce.detect_contradictions(claims)
    stability = ce.score_stability(G)

    mem_after = _measure_memory()
    elapsed = time.time() - start_time

    return {
        "agents": num_agents,
        "edges": G.number_of_edges(),
        "stability": stability,
        "time_s": round(elapsed, 3),
        "memory_mb": round(mem_after - mem_before, 2),
    }


def run_benchmarks():
    """Run multi-scale performance benchmarks."""
    _print_banner("Running Tessrax Core Benchmarks")
    results = []
    for n in [1000, 5000, 10000]:
        stats = benchmark_contradiction_engine(n)
        results.append(stats)
        print(json.dumps(stats, indent=2))
    return results


# === INTEGRATION TEST ========================================================

def test_end_to_end_integration(tmp_path):
    """Validate full Tessrax pipeline from contradiction → governance → ledger."""
    claims = [
        {"agent": "GPT", "claim": "AI must be transparent.", "type": "normative"},
        {"agent": "Claude", "claim": "AI must not be transparent.", "type": "normative"},
        {"agent": "Gemini", "claim": "AI must be transparent.", "type": "normative"},
    ]
    G, stability = ce.run_contradiction_cycle(claims, str(tmp_path / "ledger.jsonl"))
    event = gk.route(G, stability, str(tmp_path / "gov_ledger.jsonl"))

    assert 0 <= stability <= 1
    assert event.governance_lane in [lane.value for lane in gk.GovernanceLane]
    assert Path(tmp_path / "gov_ledger.jsonl").exists(), "Governance ledger must be written"


# === COMMAND-LINE ENTRY ======================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run Tessrax Core tests and benchmarks")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmark simulations")
    parser.add_argument("--pytest", action="store_true", help="Run pytest suite")
    args = parser.parse_args()

    if args.pytest:
        sys.exit(pytest.main(["-v", __file__]))

    if args.benchmarks:
        run_benchmarks()
        print("\n✅ Benchmarks complete.")
