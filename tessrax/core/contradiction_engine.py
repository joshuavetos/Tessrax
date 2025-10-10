"""
Tessrax Contradiction Engine (CE-MOD-66)
----------------------------------------

Detects and classifies contradictions across multi-agent outputs.
Implements four contradiction types:
    • Logical
    • Temporal
    • Semantic
    • Normative

Outputs a contradiction graph (networkx) and stability metrics
for downstream governance routing via governance_kernel.py.

Version: CE-MOD-66-R2
Author: Tessrax LLC
"""

from __future__ import annotations
import hashlib
import itertools
import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx


# === ENUM DEFINITIONS ========================================================

class ContradictionType(str, Enum):
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    NORMATIVE = "normative"


# === DATA CLASSES ============================================================

@dataclass
class AgentClaim:
    agent: str
    claim: str
    reasoning: str = ""
    type: str = "normative"


@dataclass
class Contradiction:
    type: ContradictionType
    statement_a: str
    statement_b: str
    reason: str
    confidence: float


# === CORE ENGINE =============================================================

def detect_contradictions(agent_claims: List[Dict[str, str]]) -> nx.Graph:
    """
    Compare all agent claims pairwise and identify contradictions.

    Args:
        agent_claims: list of dicts with keys {agent, claim, type}

    Returns:
        A networkx.Graph object with agents as nodes and contradictions as edges.
    """
    G = nx.Graph()
    for claim in agent_claims:
        G.add_node(claim["agent"], claim=claim["claim"], type=claim["type"])

    for a, b in itertools.combinations(agent_claims, 2):
        contradiction = _compare_claims(a, b)
        if contradiction:
            G.add_edge(
                a["agent"],
                b["agent"],
                **asdict(contradiction)
            )

    return G


def _compare_claims(a: Dict[str, str], b: Dict[str, str]) -> Contradiction | None:
    """Internal helper to classify contradiction type and confidence."""
    if a["claim"] == b["claim"]:
        return None

    claim_a, claim_b = a["claim"].lower(), b["claim"].lower()
    reasoning = f"Conflict between '{claim_a}' and '{claim_b}'."

    if "must" in claim_a and "must not" in claim_b:
        ctype = ContradictionType.LOGICAL
        conf = 0.95
    elif "now" in claim_a and "later" in claim_b:
        ctype = ContradictionType.TEMPORAL
        conf = 0.85
    elif any(term in claim_a for term in ["define", "consider", "term"]) or \
         any(term in claim_b for term in ["define", "consider", "term"]):
        ctype = ContradictionType.SEMANTIC
        conf = 0.75
    else:
        ctype = ContradictionType.NORMATIVE
        conf = 0.65

    return Contradiction(
        type=ctype,
        statement_a=a["claim"],
        statement_b=b["claim"],
        reason=reasoning,
        confidence=conf
    )


# === STABILITY SCORING =======================================================

def score_stability(G: nx.Graph) -> float:
    """
    Compute a stability index (0–1) based on contradiction density.

    Returns:
        float: stability index (higher = more agreement)
    """
    if G.number_of_nodes() <= 1:
        return 1.0

    contradictions = G.number_of_edges()
    possible_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = contradictions / possible_edges if possible_edges > 0 else 0

    stability_index = round(1.0 - density, 3)
    return max(0.0, min(1.0, stability_index))


# === LEDGER LOGGING ==========================================================

def log_to_ledger(G: nx.Graph, stability: float, path: str = "data/ledger.jsonl") -> str:
    """
    Append contradiction snapshot to the ledger with hash chain integrity.

    Returns:
        str: new hash for the appended record
    """
    Path(path).parent.mkdir(exist_ok=True)
    prev_hash = _get_last_hash(path)

    record = {
        "agents": list(G.nodes()),
        "contradictions": G.number_of_edges(),
        "stability_index": stability,
        "prev_hash": prev_hash,
    }
    record_str = json.dumps(record, sort_keys=True)
    new_hash = hashlib.sha256(record_str.encode()).hexdigest()
    record["hash"] = new_hash

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return new_hash


def _get_last_hash(path: str) -> str:
    """Retrieve hash from last ledger line."""
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


# === ENTRY POINT =============================================================

def run_contradiction_cycle(agent_claims: List[Dict[str, str]], ledger_path: str = "data/ledger.jsonl") -> Tuple[nx.Graph, float]:
    """
    Full pipeline:
        1. Detect contradictions
        2. Compute stability
        3. Append to ledger
    """
    G = detect_contradictions(agent_claims)
    stability = score_stability(G)
    log_to_ledger(G, stability, ledger_path)
    return G, stability
