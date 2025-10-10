"""
Tessrax Contradiction Engine (CE-MOD-66-R3)
Adds safe regex handling to prevent ReDoS via charter rules.
"""

import itertools, json, hashlib, signal, regex as re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx


# === SAFE REGEX =============================================================

class TimeoutException(Exception): pass
def _timeout_handler(signum, frame): raise TimeoutException()

def safe_match(pattern: str, text: str, timeout: float = 0.2):
    """Perform regex search with timeout protection."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return re.search(pattern, text)
    except TimeoutException:
        raise RuntimeError(f"Regex timeout on pattern: {pattern[:40]}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


# === ENUMS / DATA CLASSES ====================================================

class ContradictionType(str, Enum):
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    NORMATIVE = "normative"


@dataclass
class Contradiction:
    type: ContradictionType
    statement_a: str
    statement_b: str
    reason: str
    confidence: float


# === DETECTION ===============================================================

def detect_contradictions(agent_claims: List[Dict[str, str]]) -> nx.Graph:
    G = nx.Graph()
    for claim in agent_claims:
        G.add_node(claim["agent"], claim=claim["claim"], type=claim.get("type", "normative"))

    for a, b in itertools.combinations(agent_claims, 2):
        c = _compare_claims(a, b)
        if c:
            G.add_edge(a["agent"], b["agent"], **asdict(c))
    return G


def _compare_claims(a: Dict[str, str], b: Dict[str, str]) -> Contradiction | None:
    if a["claim"] == b["claim"]:
        return None

    ca, cb = a["claim"].lower(), b["claim"].lower()
    reasoning = f"Conflict between '{ca}' and '{cb}'."

    # basic pattern checks using safe_match
    if safe_match(r"\bmust\b", ca) and safe_match(r"\bmust not\b", cb):
        ctype, conf = ContradictionType.LOGICAL, 0.95
    elif safe_match(r"\bnow\b", ca) and safe_match(r"\blater\b", cb):
        ctype, conf = ContradictionType.TEMPORAL, 0.85
    elif safe_match(r"\bdefine\b", ca) or safe_match(r"\bdefine\b", cb):
        ctype, conf = ContradictionType.SEMANTIC, 0.75
    else:
        ctype, conf = ContradictionType.NORMATIVE, 0.65

    return Contradiction(ctype, a["claim"], b["claim"], reasoning, conf)


# === STABILITY / LEDGER =====================================================

def score_stability(G: nx.Graph) -> float:
    if G.number_of_nodes() <= 1: return 1.0
    contradictions = G.number_of_edges()
    possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
    density = contradictions / possible if possible else 0
    return round(max(0.0, 1 - density), 3)


def _get_last_hash(path: str) -> str:
    if not Path(path).exists(): return "0"*64
    with open(path, "r", encoding="utf-8") as f:
        lines=f.readlines()
        if not lines: return "0"*64
        try: return json.loads(lines[-1])["hash"]
        except Exception: return "0"*64


def log_to_ledger(G: nx.Graph, stability: float,
                  path: str = "data/ledger.jsonl") -> str:
    Path(path).parent.mkdir(exist_ok=True)
    prev_hash = _get_last_hash(path)
    record = {
        "agents": list(G.nodes()),
        "contradictions": G.number_of_edges(),
        "stability_index": stability,
        "prev_hash": prev_hash,
    }
    s = json.dumps(record, sort_keys=True)
    record["hash"] = hashlib.sha256(s.encode()).hexdigest()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record)+"\n")
    return record["hash"]


def run_contradiction_cycle(agent_claims: List[Dict[str,str]],
                            ledger_path: str="data/ledger.jsonl") -> Tuple[nx.Graph,float]:
    G = detect_contradictions(agent_claims)
    stability = score_stability(G)
    log_to_ledger(G, stability, ledger_path)
    return G, stability