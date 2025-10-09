"""
Unit tests for CE-MOD-66 Contradiction Graph Engine
Covers: empty input, single agent, perfect consensus, full disagreement, large synthetic dataset.
"""
import json
import pytest
import networkx as nx
from ce_mod_66 import detect_contradictions, score_stability


def test_empty_claims():
    claims = []
    G = detect_contradictions(claims)
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == 0
    assert score_stability(G) == 1.0  # No edges → perfect stability


def test_single_agent():
    claims = [{"agent": "Solo", "claim": "Option A", "type": "normative"}]
    G = detect_contradictions(claims)
    assert len(G.nodes) == 1
    assert len(G.edges) == 0
    assert score_stability(G) == 1.0  # No contradictions possible


def test_perfect_consensus():
    claims = [
        {"agent": "A", "claim": "Yes", "type": "normative"},
        {"agent": "B", "claim": "Yes", "type": "normative"},
        {"agent": "C", "claim": "Yes", "type": "normative"}
    ]
    G = detect_contradictions(claims)
    assert all(not d.get("contradiction") for _, _, d in G.edges(data=True))
    assert round(score_stability(G), 2) == 1.00


def test_total_disagreement():
    claims = [
        {"agent": "A", "claim": "X", "type": "normative"},
        {"agent": "B", "claim": "Y", "type": "normative"},
        {"agent": "C", "claim": "Z", "type": "normative"}
    ]
    G = detect_contradictions(claims)
    contradictions = sum(1 for _, _, d in G.edges(data=True) if d.get("contradiction"))
    assert contradictions == len(G.edges)
    assert round(score_stability(G), 2) == 0.00


def test_large_synthetic_benchmark(benchmark):
    # Generate 1,000 synthetic agent claims with 10 claim categories
    import random
    random.seed(42)
    claims = [
        {"agent": f"Agent_{i}", "claim": f"Option_{random.randint(1,10)}", "type": "normative"}
        for i in range(1000)
    ]
    def run_detection():
        G = detect_contradictions(claims)
        return score_stability(G)
    stability = benchmark(run_detection)
    assert 0.0 <= stability <= 1.0