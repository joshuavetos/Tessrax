#!/usr/bin/env python3
"""
Tessrax Core Engine — Unified Contradiction + Graph Module
----------------------------------------------------------
Merged modules:
  • contradiction_engine.py
  • ce-mod-66.py
  • conflict_graph.py

Purpose:
  Detect and score contradictions between agent claims, construct a semantic graph,
  compute stability metrics, and prepare structured governance-ready results.
"""

import hashlib
import json
from collections import defaultdict
import networkx as nx
from typing import List, Dict, Any, Tuple


# ----------------------------------------------------------------------
# Core Data Structures
# ----------------------------------------------------------------------

class Contradiction:
    """Represents a single contradiction between two agent claims."""
    def __init__(self, agent_a, claim_a, agent_b, claim_b, reason, ctype):
        self.agent_a = agent_a
        self.claim_a = claim_a
        self.agent_b = agent_b
        self.claim_b = claim_b
        self.reason = reason
        self.ctype = ctype

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_a": self.agent_a,
            "claim_a": self.claim_a,
            "agent_b": self.agent_b,
            "claim_b": self.claim_b,
            "reason": self.reason,
            "type": self.ctype,
        }


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def _hash_text(text: str) -> str:
    """Return SHA256 hash of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_claim(text: str) -> str:
    """Lightweight normalization for comparing claims."""
    return text.strip().lower().replace(".", "").replace(",", "")


# ----------------------------------------------------------------------
# Contradiction Detection Engine
# ----------------------------------------------------------------------

def detect_contradictions(agent_claims: List[Dict[str, str]]) -> nx.Graph:
    """
    Detect contradictions among agent claims and return a contradiction graph.

    Each node = agent.
    Each edge = detected contradiction with metadata.
    """
    G = nx.Graph()
    for claim in agent_claims:
        G.add_node(claim["agent"], claim=claim["claim"], type=claim.get("type", "unknown"))

    for i, a in enumerate(agent_claims):
        for j, b in enumerate(agent_claims):
            if j <= i:
                continue

            claim_a = _normalize_claim(a["claim"])
            claim_b = _normalize_claim(b["claim"])

            # Simple contradiction rule: opposing claims (A vs B, yes vs no, etc.)
            if (claim_a != claim_b) and not claim_a.startswith(claim_b) and not claim_b.startswith(claim_a):
                reason = f"Conflict between '{a['claim']}' and '{b['claim']}'"
                contradiction = Contradiction(a["agent"], a["claim"], b["agent"], b["claim"], reason, "logical")
                G.add_edge(a["agent"], b["agent"], data=contradiction.to_dict())

    return G


# ----------------------------------------------------------------------
# CE-MOD-66: Enhanced Conflict Graph Scoring
# ----------------------------------------------------------------------

def score_stability(G: nx.Graph) -> float:
    """
    Compute a stability score based on contradiction density.
    Lower contradiction density → higher stability.
    """
    if G.number_of_nodes() == 0:
        return 1.0
    if G.number_of_edges() == 0:
        return 1.0

    edge_factor = G.number_of_edges() / (G.number_of_nodes() ** 2)
    stability = max(0.0, 1.0 - edge_factor * 4.0)  # 4x exaggeration for visible contrast
    return round(stability, 3)


def classify_governance_lane(stability: float) -> str:
    """
    Assign governance lane based on stability score.
    """
    if stability >= 0.80:
        return "Autonomic"
    elif stability >= 0.50:
        return "Deliberative"
    elif stability >= 0.25:
        return "Constitutional"
    else:
        return "Behavioral Audit"


def summarize_graph(G: nx.Graph) -> Dict[str, Any]:
    """
    Produce a structured summary of the graph analysis.
    """
    stability = score_stability(G)
    lane = classify_governance_lane(stability)
    contradictions = [
        G[u][v]["data"] for u, v in G.edges()
        if "data" in G[u][v]
    ]
    return {
        "agents": list(G.nodes()),
        "num_agents": G.number_of_nodes(),
        "num_contradictions": len(contradictions),
        "stability_index": stability,
        "governance_lane": lane,
        "contradictions": contradictions,
    }


# ----------------------------------------------------------------------
# Export / Integration
# ----------------------------------------------------------------------

def export_governance_case(G: nx.Graph, case_id: str = None) -> Dict[str, Any]:
    """
    Convert the analyzed graph into a governance-ready JSON case.
    """
    summary = summarize_graph(G)
    return {
        "case_id": case_id or _hash_text(json.dumps(summary))[:8],
        "description": f"Contradiction analysis for {summary['num_agents']} agents",
        "stability_index": summary["stability_index"],
        "lane": summary["governance_lane"],
        "reason": f"{summary['num_contradictions']} contradictions detected among agents",
        "agents": summary["agents"],
        "contradictions": summary["contradictions"],
    }


# ----------------------------------------------------------------------
# CLI / Demo Entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    demo_claims = [
        {"agent": "GPT", "claim": "A", "type": "normative"},
        {"agent": "Claude", "claim": "B", "type": "normative"},
        {"agent": "Gemini", "claim": "A", "type": "normative"},
    ]

    print("\n--- Tessrax Engine Demo ---")
    graph = detect_contradictions(demo_claims)
    result = export_governance_case(graph)
    print(json.dumps(result, indent=2))
