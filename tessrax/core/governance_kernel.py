"""
Tessrax Governance Kernel (GK-MOD-01-R3)
----------------------------------------
Adds optional Redis-based distributed locking for scalable consensus.
If REDIS_URL env var is set, uses Redis; otherwise falls back to file locks.
"""

import os, json, hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import networkx as nx
from filelock import FileLock

try:
    from redis import Redis
except ImportError:
    Redis = None


# === ENUMS ==================================================================

class GovernanceLane(str, Enum):
    AUTONOMIC = "Autonomic"
    DELIBERATIVE = "Deliberative"
    CONSTITUTIONAL = "Constitutional"
    BEHAVIORAL_AUDIT = "Behavioral Audit"


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


# === DISTRIBUTED LOCK ========================================================

def get_lock(domain: str):
    """Return either a Redis or file-based lock depending on environment."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url and Redis:
        r = Redis.from_url(redis_url)
        return r.lock(f"tessrax:{domain}", timeout=10)
    return FileLock(f"/tmp/{domain}.lock")


# === CLASSIFICATION ==========================================================

def classify_lane(G: nx.Graph, stability_index: float) -> GovernanceLane:
    for _, _, data in G.edges(data=True):
        if data.get("type", "").lower() == "semantic":
            return GovernanceLane.BEHAVIORAL_AUDIT
    if stability_index >= 0.75:
        return GovernanceLane.AUTONOMIC
    elif 0.40 <= stability_index < 0.75:
        return GovernanceLane.DELIBERATIVE
    return GovernanceLane.CONSTITUTIONAL


def summarize(G: nx.Graph, stability_index: float) -> str:
    c = G.number_of_edges()
    if c == 0:
        return "No contradictions detected; consensus stable."
    if stability_index < 0.40:
        return "Severe contradiction density; potential rule drift."
    if stability_index < 0.75:
        return "Moderate disagreement; deliberation recommended."
    return "High stability; safe for auto-adoption."


# === LEDGER =================================================================

def _get_last_hash(path: str) -> str:
    if not Path(path).exists():
        return "0"*64
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return "0"*64
        try:
            return json.loads(lines[-1])["hash"]
        except Exception:
            return "0"*64


def record_event(G: nx.Graph, stability_index: float,
                 path: str = "data/governance_ledger.jsonl") -> GovernanceEvent:
    prev_hash = _get_last_hash(path)
    lane = classify_lane(G, stability_index)
    summary = summarize(G, stability_index)

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
    event["hash"] = hashlib.sha256(event_str.encode()).hexdigest()

    Path(path).parent.mkdir(exist_ok=True)
    with get_lock("governance_ledger"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    return GovernanceEvent(**event)


def route(G: nx.Graph, stability_index: float,
          ledger_path: str = "data/governance_ledger.jsonl") -> GovernanceEvent:
    """Public entry point with automatic locking."""
    with get_lock("governance_route"):
        return record_event(G, stability_index, ledger_path)