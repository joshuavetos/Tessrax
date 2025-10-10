"""
Tessrax Governance Kernel (GK-MOD-01)
-------------------------------------

Core routing layer for Tessrax contradiction metabolism.
Consumes contradiction graphs from CE-MOD-66 and produces
structured governance events for the ledger.

Implements the four-lane governance model:

    • Autonomic       — Consensus stable (>0.75)
    • Deliberative    — Moderate disagreement (0.40–0.75)
    • Constitutional  — Foundational contradiction (<0.40)
    • BehavioralAudit — Semantic or definitional manipulation detected

Each decision is logged immutably with hash chaining for auditability.

Version: GK-MOD-01-R2
Author: Tessrax LLC
"""

from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx


# === ENUM DEFINITIONS ========================================================

class GovernanceLane(str, Enum):
    AUTONOMIC = "Autonomic"
    DELIBERATIVE = "Deliberative"
    CONSTITUTIONAL = "Constitutional"
    BEHAVIORAL_AUDIT = "Behavioral Audit"


# === DATA CLASSES ============================================================

@dataclass
class GovernanceEvent:
    timestamp: str
    agents: List[str]
    stability_index: float
    governance_lane: GovernanceLane
    contradictions: int
    note: str
    prev_hash: str
    hash: str


# === KERNEL LOGIC ============================================================

def classify_lane(G: nx.Graph, stability_index: float) -> GovernanceLane:
    """
    Classify the contradiction graph into a governance lane.
    """
    # Case 1: Detect semantic manipulation
    for _, _, data in G.edges(data=True):
        if data.get("type", "").lower() == "semantic":
            return GovernanceLane.BEHAVIORAL_AUDIT

    # Case 2: Threshold-based routing
    if stability_index >= 0.75:
        return GovernanceLane.AUTONOMIC
    elif 0.40 <= stability_index < 0.75:
        return GovernanceLane.DELIBERATIVE
    else:
        return GovernanceLane.CONSTITUTIONAL


def summarize(G: nx.Graph, stability_index: float) -> Dict[str, str]:
    """
    Generate a short diagnostic summary note.
    """
    contradictions = G.number_of_edges()

    if contradictions == 0:
        return {"note": "No contradictions detected; consensus stable."}
    elif stability_index < 0.40:
        return {"note": "Severe contradiction density detected; potential rule drift."}
    elif stability_index < 0.75:
        return {"note": "Moderate disagreement; deliberation recommended."}
    else:
        return {"note": "High stability; safe for auto-adoption."}


# === LEDGER CHAIN ============================================================

def _get_last_hash(path: str) -> str:
    """Retrieve last ledger hash."""
    if not Path(path).exists():
        return "0" * 64
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return "0" * 64
        try:
            return json.loads(lines[-1])["hash"]
        except Exception:
            return "0" * 64


def record_event(G: nx.Graph, stability_index: float, path: str = "data/governance_ledger.jsonl") -> GovernanceEvent:
    """
    Record a new governance event to the ledger.
    """
    from datetime import datetime
    prev_hash = _get_last_hash(path)
    lane = classify_lane(G, stability_index)
    summary = summarize(G, stability_index)["note"]

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agents": list(G.nodes()),
        "stability_index": stability_index,
        "governance_lane": lane.value,
        "contradictions": G.number_of_edges(),
        "note": summary,
        "prev_hash": prev_hash,
    }

    event_str = json.dumps(event, sort_keys=True)
    new_hash = hashlib.sha256(event_str.encode()).hexdigest()
    event["hash"] = new_hash

    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return GovernanceEvent(**event)


# === KERNEL INTERFACE ========================================================

def route(G: nx.Graph, stability_index: float, ledger_path: str = "data/governance_ledger.jsonl") -> GovernanceEvent:
    """
    High-level routing wrapper.

    Args:
        G: Contradiction graph
        stability_index: Stability value from CE-MOD-66
        ledger_path: File path for governance ledger

    Returns:
        GovernanceEvent dataclass
    """
    return record_event(G, stability_index, ledger_path)


def analyze_and_route(agent_claims: List[Dict[str, str]], contradiction_engine) -> GovernanceEvent:
    """
    End-to-end route:
        1. Detect contradictions
        2. Score stability
        3. Classify governance lane
        4. Append event to ledger
    """
    G, stability = contradiction_engine.run_contradiction_cycle(agent_claims)
    event = route(G, stability)
    return event
