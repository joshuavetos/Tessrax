"""
Tessrax Engine Core
Unified contradiction + governance orchestrator.

Combines logic from:
- contradiction_engine.py
- ce_mod_66.py
- governance_kernel.py
"""

import json
import itertools
import networkx as nx
from datetime import datetime
from pathlib import Path

# === CONTRADICTION GRAPH (formerly ce_mod_66) ===

def detect_contradictions(agent_claims):
    """Build a contradiction graph from agent claim dictionaries."""
    G = nx.Graph()
    for a in agent_claims:
        G.add_node(a["agent"], claim=a["claim"], ctype=a.get("type", "unspecified"))
    for a, b in itertools.combinations(agent_claims, 2):
        G.add_edge(a["agent"], b["agent"], contradiction=(a["claim"] != b["claim"]))
    return G

def score_stability(G):
    """Compute stability index (S = 1 - contradictions / total_edges)."""
    total_edges = len(G.edges)
    if total_edges == 0:
        return 1.0
    contradictions = sum(1 for _, _, d in G.edges(data=True) if d.get("contradiction"))
    return 1 - (contradictions / total_edges)

# === GOVERNANCE KERNEL (formerly governance_kernel.py) ===

def route(G):
    """Route a contradiction graph to the appropriate governance lane."""
    stability = score_stability(G)
    if stability > 0.9:
        lane = "autonomic"
    elif stability > 0.7:
        lane = "deliberative"
    elif stability > 0.5:
        lane = "constitutional"
    else:
        lane = "behavioral_audit"
    return {"stability": round(stability, 3), "lane": lane}

# === ORCHESTRATION ===

def analyze(agent_claims, ledger_path="data/ledger.jsonl"):
    """Full analysis pipeline: build graph → route → append result to ledger."""
    G = detect_contradictions(agent_claims)
    result = route(G)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stability_index": result["stability"],
        "governance_lane": result["lane"],
        "agents": [a["agent"] for a in agent_claims],
    }
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    return result

if __name__ == "__main__":
    sample = [
        {"agent": "GPT", "claim": "Option A"},
        {"agent": "Gemini", "claim": "Option B"},
        {"agent": "Claude", "claim": "Option A"},
    ]
    print(analyze(sample))
