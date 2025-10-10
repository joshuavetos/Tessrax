#!/usr/bin/env python3
"""
Tessrax Test Engine — Unified Benchmark + Reliability Harness
--------------------------------------------------------------
Merged modules:
  • benchmark_runner.py
  • reliability_harness.py
  • agent_validation_harness.py

Purpose:
  Validate and benchmark Tessrax Core across multiple dimensions:
  - Performance (time + memory)
  - Consistency (reproducible contradiction detection)
  - Reliability (no crashes or data corruption)
"""

import json
import os
import random
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any
from tessrax.core.engine import detect_contradictions, score_stability
from tessrax.core.ledger import append_entry, verify_chain


# ----------------------------------------------------------------------
# Synthetic Data Generator
# ----------------------------------------------------------------------

def generate_claims(n: int, contradiction_rate: float = 0.5) -> List[Dict[str, Any]]:
    """
    Generate synthetic agent claims for testing.

    Args:
        n: Number of agents
        contradiction_rate: Proportion of agents to assign opposite claims
    """
    claims = []
    split_point = int(n * contradiction_rate)
    for i in range(n):
        claim_value = "A" if i < split_point else "B"
        claims.append({"agent": f"Agent-{i}", "claim": claim_value, "type": "normative"})
    return claims


# ----------------------------------------------------------------------
# Benchmark Suite
# ----------------------------------------------------------------------

def run_benchmark(sizes: List[int] = [100, 1000, 5000, 10000]) -> List[Dict[str, Any]]:
    """
    Measure performance (time and memory) for contradiction detection.
    """
    results = []
    for n in sizes:
        claims = generate_claims(n)
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        t0 = time.time()
        G = detect_contradictions(claims)
        stability = score_stability(G)
        t1 = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024
        delta_t = round(t1 - t0, 4)
        delta_mem = round(mem_after - mem_before, 2)

        results.append({
            "agents": n,
            "time_sec": delta_t,
            "memory_mb": delta_mem,
            "stability_index": stability
        })
        print(f"[Benchmark] {n} agents → {delta_t:.3f}s, {delta_mem:.2f}MB, stability={stability}")
    return results


# ----------------------------------------------------------------------
# Reliability Tests
# ----------------------------------------------------------------------

def test_repeatability(n: int = 500) -> Dict[str, Any]:
    """
    Verify contradiction detection produces identical results across runs.
    """
    claims = generate_claims(n)
    runs = [score_stability(detect_contradictions(claims)) for _ in range(3)]
    consistent = all(abs(runs[0] - r) < 1e-9 for r in runs)
    result = {
        "consistent": consistent,
        "stabilities": runs
    }
    print(f"[Repeatability] Stable across runs: {consistent} ({runs})")
    return result


def test_integrity_recovery() -> Dict[str, Any]:
    """
    Verify that the ledger chain verification detects and isolates corruption.
    """
    ok, break_line = verify_chain()
    result = {
        "ledger_intact": ok,
        "break_line": break_line if not ok else None
    }
    print(f"[Integrity] Ledger valid={ok} (break_line={break_line})")
    return result


# ----------------------------------------------------------------------
# Multi-Agent Validation
# ----------------------------------------------------------------------

def validate_agent_distribution(agent_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure balanced representation of claims among agents.
    """
    totals = {"A": 0, "B": 0}
    for c in agent_claims:
        if c["claim"] in totals:
            totals[c["claim"]] += 1
    ratio = totals["A"] / max(1, (totals["A"] + totals["B"]))
    balance = abs(0.5 - ratio)
    result = {"balance_offset": round(balance, 3), "distribution": totals}
    print(f"[Validation] Distribution: {totals}, offset={balance}")
    return result


def simulate_agent_noise(agent_claims: List[Dict[str, Any]], noise_rate: float = 0.05) -> List[Dict[str, Any]]:
    """
    Introduce random noise (claim flips) into agent dataset.
    """
    mutated = []
    for claim in agent_claims:
        new_claim = claim["claim"]
        if random.random() < noise_rate:
            new_claim = "B" if new_claim == "A" else "A"
        mutated.append({"agent": claim["agent"], "claim": new_claim, "type": "normative"})
    print(f"[Noise] Introduced noise into {int(noise_rate*100)}% of claims.")
    return mutated


# ----------------------------------------------------------------------
# Full Test Harness
# ----------------------------------------------------------------------

def run_full_test_suite() -> Dict[str, Any]:
    """
    Run all validation, reliability, and benchmark tests together.
    """
    print("\n--- Running Tessrax Core Test Suite ---\n")
    results = {
        "benchmarks": run_benchmark(),
        "repeatability": test_repeatability(),
        "integrity": test_integrity_recovery()
    }

    sample = generate_claims(200)
    results["validation"] = validate_agent_distribution(sample)
    noisy = simulate_agent_noise(sample)
    results["validation_after_noise"] = validate_agent_distribution(noisy)

    Path("data/reports").mkdir(parents=True, exist_ok=True)
    report_file = Path("data/reports/test_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Report] Saved to {report_file}")
    return results


# ----------------------------------------------------------------------
# CLI Entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    final = run_full_test_suite()
    print("\n--- Summary ---")
    print(json.dumps(final, indent=2))
