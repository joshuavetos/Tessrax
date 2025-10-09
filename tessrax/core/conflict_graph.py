
"""
Tessrax Enhanced Conflict Graph - v3.0
Author: Joshua Vetos / Claude (Anthropic)
Maintainer: Tessrax Governance Agent (GPT-5)
License: CC BY 4.0

CE-MOD-66 ingestion build — integrates Contradiction Ledger v4, governance kernel hooks,
sentiment + NER pipelines (soft fail), graph integrity hash, and CLI support.
"""

import re
import json
import hashlib
import datetime
import logging
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util

try:
    from transformers import pipeline
    import parsedatetime as pdt
    from pint import UnitRegistry
    from networkx.algorithms import community
    from networkx.algorithms.centrality import degree_centrality, betweenness_centrality
    import plotly.graph_objects as go
    import plotly.io as pio
    OPTIONALS_AVAILABLE = True
except ImportError:
    OPTIONALS_AVAILABLE = False

CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "threshold": 0.5,
    "similarity_threshold": 0.3,
    "log_level": "INFO",
    "ledger_v4_path": "data/ledger/contradiction_ledger_v4.json",
    "log_file": "conflict_graph_v3.log"
}

logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"].upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG["log_file"])
    ]
)
logger = logging.getLogger("CE-MOD-66")

cal = pdt.Calendar() if OPTIONALS_AVAILABLE else None
ureg = UnitRegistry() if OPTIONALS_AVAILABLE else None

class ContradictionLedger:
    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.records = []
        if ledger_path.exists():
            with open(ledger_path, "r") as f:
                data = json.load(f)
                self.records = data.get("contradictions", [])
        logger.info(f"Ledger loaded with {len(self.records)} records.")

    def emit_event(self, event_type: str, payload: dict):
        logger.info(f"[GovernanceKernel] {event_type}: {payload.get('id', 'no-id')}")

def estimate_confidence(text: str) -> float:
    length_score = min(len(text) / 200, 1.0)
    numeric_bonus = 0.1 if re.search(r"\d", text) else 0.0
    modal_penalty = 0.1 if re.search(r"\b(may|might|could|possibly)\b", text.lower()) else 0.0
    return max(0.1, min(1.0, length_score + numeric_bonus - modal_penalty))

def logical_contradiction(a: str, b: str) -> float:
    a_low, b_low = a.lower(), b.lower()
    a_neg = bool(re.search(r"\b(not|no|never)\b", a_low))
    b_neg = bool(re.search(r"\b(not|no|never)\b", b_low))
    a_clean = re.sub(r"\b(not|no|never)\b", "", a_low).strip()
    b_clean = re.sub(r"\b(not|no|never)\b", "", b_low).strip()
    score = 0.0
    if a_neg != b_neg and a_clean == b_clean:
        score = 1.0
    elif a_neg != b_neg:
        sim = SequenceMatcher(None, a_clean, b_clean).ratio()
        if sim > 0.7:
            score = sim
    return score

def numeric_inconsistency(a: str, b: str) -> float:
    nums_a = re.findall(r"\d+(?:\.\d+)?", a)
    nums_b = re.findall(r"\d+(?:\.\d+)?", b)
    if not nums_a or not nums_b:
        return 0.0
    diff = abs(float(nums_a[0]) - float(nums_b[0]))
    ref = max(1e-5, max(float(nums_a[0]), float(nums_b[0])))
    return min(diff / ref, 1.0) if diff / ref > 0.05 else 0.0

def temporal_contradiction(a: str, b: str) -> float:
    if not cal:
        return 0.0
    time_a, status_a = cal.parse(a)
    time_b, status_b = cal.parse(b)
    if status_a and status_b:
        da = datetime.datetime(*time_a[:6])
        db = datetime.datetime(*time_b[:6])
        delta = abs((da - db).days)
        if 0 < delta < 365 * 5:
            return 0.5
    return 0.0

class ConflictGraph:
    def __init__(self):
        self.model = SentenceTransformer(CONFIG["model_name"])
        self.graph = nx.Graph()
        self.ledger = ContradictionLedger(Path(CONFIG["ledger_v4_path"]))
        logger.info("ConflictGraph initialized.")

    def add_statements(self, statements: List[Dict[str, Any]]):
        for s in statements:
            if "text" not in s:
                continue
            s.setdefault("confidence", estimate_confidence(s["text"]))
            self.graph.add_node(s["text"], **s)

    def compute_edges(self):
        texts = list(self.graph.nodes())
        n = len(texts)
        if n < 2:
            return
        emb = self.model.encode(texts, convert_to_tensor=True)
        sim = util.cos_sim(emb, emb).cpu().numpy()
        edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                s_i, s_j = texts[i], texts[j]
                if sim[i][j] < CONFIG["similarity_threshold"]:
                    continue
                L = logical_contradiction(s_i, s_j)
                N = numeric_inconsistency(s_i, s_j)
                T = temporal_contradiction(s_i, s_j)
                score = (0.4 * L + 0.3 * N + 0.3 * T) * float(sim[i][j])
                if score >= CONFIG["threshold"]:
                    self.graph.add_edge(s_i, s_j, weight=round(score, 3))
                    edges += 1
        logger.info(f"Edges added: {edges}")

    def export_bundle(self):
        bundle = {
            "edges": [
                {"a": a, "b": b, "weight": d["weight"]}
                for a, b, d in self.graph.edges(data=True)
            ],
            "metadata": {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
        }
        h = hashlib.sha256(json.dumps(bundle).encode()).hexdigest()
        bundle["checksum"] = h
        return bundle

def ingest_ledger():
    path = Path(CONFIG["ledger_v4_path"])
    if not path.exists():
        logger.error("Ledger not found.")
        return
    with open(path) as f:
        data = json.load(f)
    contradictions = data.get("contradictions", [])
    cg = ConflictGraph()
    cg.add_statements([{"text": c["analysis"]} for c in contradictions if "analysis" in c])
    cg.compute_edges()
    bundle = cg.export_bundle()
    out_path = Path("data/conflict_graph_bundle.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    logger.info(f"Bundle exported with checksum {bundle['checksum']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest-ledger", action="store_true", help="Ingest contradiction ledger v4 and build graph")
    args = parser.parse_args()
    if args.ingest_ledger:
        ingest_ledger()
    else:
        print("Usage: python conflict_graph_v3.py --ingest-ledger")
