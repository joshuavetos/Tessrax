“””
Tessrax Enhanced Conflict Graph - v2.0
Author: Joshua Vetos / Claude (Anthropic)
License: CC BY 4.0

Complete implementation with sentiment analysis, NER, advanced graph analysis,
enhanced contradiction metrics, interactive visualization, and comprehensive logging.
“””

import re
import json
import hashlib
import datetime
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import parsedatetime as pdt
from pint import UnitRegistry

try:
from networkx.algorithms import community
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality
from networkx.algorithms.shortest_paths import shortest_path
NETWORKX_ADVANCED = True
except ImportError:
NETWORKX_ADVANCED = False

try:
import plotly.graph_objects as go
import plotly.io as pio
PLOTLY_AVAILABLE = True
except ImportError:
PLOTLY_AVAILABLE = False

try:
import matplotlib.pyplot as plt
MATPLOTLIB_AVAILABLE = True
except ImportError:
MATPLOTLIB_AVAILABLE = False

# ============================================================

# CONFIGURATION

# ============================================================

CONFIG = {
“model_name”: “all-MiniLM-L6-v2”,
“threshold”: 0.5,
“ledger_db_path”: “data/conflict_graph_ledger.db”,
“log_level”: “INFO”,
“log_file”: “conflict_graph.log”,
“similarity_threshold”: 0.3,
“contradiction_weights”: {
“logical”: 0.4,
“numeric”: 0.3,
“temporal”: 0.2,
“categorical”: 0.1
}
}

# ============================================================

# LOGGING SETUP

# ============================================================

logging.basicConfig(
level=getattr(logging, CONFIG[“log_level”].upper(), logging.INFO),
format=’%(asctime)s - %(name)s - %(levelname)s - %(message)s’,
handlers=[
logging.StreamHandler(sys.stdout),
logging.FileHandler(CONFIG[“log_file”])
]
)
logger = logging.getLogger(**name**)

# ============================================================

# EXTERNAL LIBRARIES INITIALIZATION

# ============================================================

cal = pdt.Calendar()
ureg = UnitRegistry()

# ============================================================

# DUMMY CONTRADICTION LEDGER

# ============================================================

class ContradictionLedger:
“”“Mock ledger for demonstration purposes.”””

```
def __init__(self, db_path: Path):
    self.db_path = db_path
    self.records = []
    logger.info(f"ContradictionLedger initialized: {db_path}")

def record_contradiction(self, claim_a: Dict, claim_b: Dict, 
                       resolution: str, metadata: Dict) -> Dict:
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "claim_a": claim_a,
        "claim_b": claim_b,
        "resolution": resolution,
        "metadata": metadata
    }
    self.records.append(record)
    return {"status": "success", "record_id": len(self.records)}

def close(self):
    logger.info("ContradictionLedger closed")
```

# ============================================================

# UTILITY FUNCTIONS

# ============================================================

def estimate_confidence(text: str) -> float:
“”“Heuristic confidence score based on text characteristics.”””
length_score = min(len(text) / 200, 1.0)
numeric_bonus = 0.1 if re.search(r”\d”, text) else 0.0
modal_penalty = 0.1 if re.search(
r”\b(may|might|could|possibly)\b”, text.lower()
) else 0.0
return max(0.1, min(1.0, length_score + numeric_bonus - modal_penalty))

# ============================================================

# ENHANCED CONTRADICTION METRICS

# ============================================================

def logical_contradiction(a: str, b: str,
sentiment_a: Optional[Dict],
sentiment_b: Optional[Dict]) -> float:
“”“Enhanced negation detection with sentiment analysis.”””
a_low, b_low = a.lower(), b.lower()

```
# Negation detection
a_negated = bool(re.search(r"\b(not|no|never)\b", a_low))
b_negated = bool(re.search(r"\b(not|no|never)\b", b_low))

# Core content extraction
a_clean = re.sub(r"\b(not|no|never)\b", "", a_low).strip()
b_clean = re.sub(r"\b(not|no|never)\b", "", b_low).strip()

score = 0.0

# Exact negation match
if a_negated != b_negated and a_clean == b_clean:
    score = 1.0
# Partial negation match
elif a_negated != b_negated:
    similarity = SequenceMatcher(None, a_clean, b_clean).ratio()
    if similarity > 0.7:
        score = similarity

# Sentiment enhancement
if sentiment_a and sentiment_b:
    if (sentiment_a['label'] != sentiment_b['label'] and 
        sentiment_a['score'] > 0.8 and sentiment_b['score'] > 0.8):
        score = min(1.0, score + 0.1)

return score
```

def numeric_inconsistency(a: str, b: str,
ner_a: Optional[List[Dict]],
ner_b: Optional[List[Dict]]) -> float:
“”“Enhanced numeric comparison with unit handling via Pint.”””
if not ner_a or not ner_b:
# Fallback to regex
nums_a = [float(x.strip(”%”)) for x in re.findall(r”\d+.?\d*%?”, a)]
nums_b = [float(x.strip(”%”)) for x in re.findall(r”\d+.?\d*%?”, b)]

```
    if not nums_a or not nums_b:
        return 0.0
    
    diff = abs(nums_a[0] - nums_b[0])
    ref = max(1e-5, max(nums_a[0], nums_b[0]))
    score = diff / ref
    return min(score, 1.0) if score > 0.1 else 0.0

# NER-based extraction
numbers_a = [e for e in ner_a if e['entity_group'] in ['CARDINAL', 'PERCENT']]
numbers_b = [e for e in ner_b if e['entity_group'] in ['CARDINAL', 'PERCENT']]

if not numbers_a or not numbers_b:
    return 0.0

try:
    # Pint-based comparison
    q_a = ureg(numbers_a[0]['word'].replace('%', ' percent'))
    q_b = ureg(numbers_b[0]['word'].replace('%', ' percent'))
    
    q_b_converted = q_b.to(q_a.units)
    diff = abs(q_a.magnitude - q_b_converted.magnitude)
    ref = max(1e-5, max(abs(q_a.magnitude), abs(q_b_converted.magnitude)))
    score = diff / ref
except:
    # Fallback to magnitude comparison
    try:
        q_a = float(numbers_a[0]['word'].replace('%', ''))
        q_b = float(numbers_b[0]['word'].replace('%', ''))
        diff = abs(q_a - q_b)
        ref = max(1e-5, max(abs(q_a), abs(q_b)))
        score = diff / ref
    except:
        return 0.0

return min(score, 1.0) if score > 0.05 else 0.0
```

def temporal_contradiction(a: str, b: str,
ner_a: Optional[List[Dict]],
ner_b: Optional[List[Dict]]) -> float:
“”“Enhanced temporal inconsistency detection with parsedatetime.”””
# Parse temporal information
time_a, status_a = cal.parse(a)
time_b, status_b = cal.parse(b)

```
date_a = datetime.datetime(*time_a[:6]) if status_a > 0 else None
date_b = datetime.datetime(*time_b[:6]) if status_b > 0 else None

score = 0.0

if date_a and date_b:
    time_diff = abs((date_a - date_b).total_seconds())
    # Potential inconsistency within 5 years
    if 0 < time_diff < (5 * 365 * 24 * 3600):
        score = 0.4
    
    # Pattern-based detection
    by_a = bool(re.search(r"\bby\s+\d{4}", a.lower()))
    by_b = bool(re.search(r"\bby\s+\d{4}", b.lower()))
    in_a = bool(re.search(r"\bin\s+\d{4}", a.lower()))
    in_b = bool(re.search(r"\bin\s+\d{4}", b.lower()))
    
    if by_a and in_b:
        years_a = [int(y) for y in re.findall(r"\bby\s+(\d{4})", a.lower())]
        years_b = [int(y) for y in re.findall(r"\bin\s+(\d{4})", b.lower())]
        if years_a and years_b and years_b[0] > years_a[0]:
            score = max(score, 0.6)

# NER enhancement
if ner_a and ner_b:
    non_temporal_a = {e['word'] for e in ner_a 
                     if e['entity_group'] not in ['DATE', 'TIME']}
    non_temporal_b = {e['word'] for e in ner_b 
                     if e['entity_group'] not in ['DATE', 'TIME']}
    
    if non_temporal_a & non_temporal_b and score > 0:
        score = min(1.0, score + 0.2)

return score
```

def categorical_mismatch(a: str, b: str,
ner_a: Optional[List[Dict]],
ner_b: Optional[List[Dict]]) -> float:
“”“Domain-based categorical clash detection.”””
domains = {
“environmental”: [“emission”, “carbon”, “co2”, “climate”, “renewable”],
“financial”: [“profit”, “revenue”, “loss”, “earnings”, “dividend”],
“social”: [“diversity”, “equality”, “workforce”, “community”],
“operational”: [“production”, “capacity”, “efficiency”, “output”]
}

```
def get_domains(text):
    text_lower = text.lower()
    return {name for name, keywords in domains.items()
            if any(k in text_lower for k in keywords)}

domains_a = get_domains(a)
domains_b = get_domains(b)

score = 0.0
if domains_a and domains_b and not (domains_a & domains_b):
    score = 0.3

# NER enhancement
if ner_a and ner_b and score > 0:
    entities_a = {e['word'] for e in ner_a}
    entities_b = {e['word'] for e in ner_b}
    if entities_a & entities_b:
        score = min(1.0, score + 0.1)

return score
```

# ============================================================

# MAIN CONFLICT GRAPH CLASS

# ============================================================

class ConflictGraph:
“”“Enhanced contradiction detection and analysis system.”””

```
def __init__(self):
    """Initialize with configuration parameters."""
    logger.info("Initializing ConflictGraph...")
    
    self.model = SentenceTransformer(CONFIG["model_name"])
    self.threshold = CONFIG["threshold"]
    self.graph = nx.Graph()
    self.ledger = ContradictionLedger(Path(CONFIG["ledger_db_path"]))
    self.metadata = {"created": datetime.datetime.utcnow().isoformat()}
    
    # Initialize NLP pipelines
    try:
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.ner_analyzer = pipeline("ner", grouped_entities=True)
        logger.info("NLP pipelines initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize NLP pipelines: {e}")
        raise
    
    logger.info(f"ConflictGraph ready (threshold={self.threshold})")

def add_statements(self, statements: List[Dict[str, Any]]):
    """Add statements with automatic sentiment and NER analysis."""
    logger.info(f"Adding {len(statements)} statements...")
    
    for s in statements:
        if "text" not in s:
            logger.warning("Skipping statement without 'text' field")
            continue
        
        # Auto-fill metadata
        s.setdefault("confidence", estimate_confidence(s["text"]))
        s.setdefault("source", "unknown")
        s.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
        
        # NLP analysis
        try:
            s["sentiment"] = self.sentiment_analyzer(s["text"])[0]
            s["named_entities"] = self.ner_analyzer(s["text"])
        except Exception as e:
            logger.error(f"NLP analysis failed for statement: {e}")
            s["sentiment"] = None
            s["named_entities"] = []
        
        self.graph.add_node(s["text"], **s)
    
    logger.info(f"Added {self.graph.number_of_nodes()} nodes total")

def compute_edges(self, verbose: bool = False):
    """Compute contradiction edges with enhanced metrics."""
    texts = list(self.graph.nodes())
    n = len(texts)
    
    if n < 2:
        logger.info("Need at least 2 statements")
        return
    
    logger.info(f"Computing embeddings for {n} statements...")
    embeddings = self.model.encode(texts, convert_to_tensor=True, 
                                  show_progress_bar=verbose)
    cos_sim = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    edges_added = 0
    weights = CONFIG["contradiction_weights"]
    
    for i in range(n):
        for j in range(i + 1, n):
            text_i, text_j = texts[i], texts[j]
            sim = float(cos_sim[i][j])
            
            if sim < CONFIG["similarity_threshold"]:
                continue
            
            # Get node attributes
            sent_i = self.graph.nodes[text_i].get("sentiment")
            sent_j = self.graph.nodes[text_j].get("sentiment")
            ner_i = self.graph.nodes[text_i].get("named_entities")
            ner_j = self.graph.nodes[text_j].get("named_entities")
            
            # Compute metrics
            L = logical_contradiction(text_i, text_j, sent_i, sent_j)
            N = numeric_inconsistency(text_i, text_j, ner_i, ner_j)
            T = temporal_contradiction(text_i, text_j, ner_i, ner_j)
            C = categorical_mismatch(text_i, text_j, ner_i, ner_j)
            
            # Weighted score
            conf_i = self.graph.nodes[text_i]["confidence"]
            conf_j = self.graph.nodes[text_j]["confidence"]
            avg_conf = (conf_i + conf_j) / 2
            
            contradiction_score = (
                weights["logical"] * L +
                weights["numeric"] * N +
                weights["temporal"] * T +
                weights["categorical"] * C
            ) * sim * avg_conf
            
            if contradiction_score >= self.threshold:
                self.graph.add_edge(
                    text_i, text_j,
                    weight=round(contradiction_score, 3),
                    sim=round(sim, 3),
                    metrics={"L": round(L, 2), "N": round(N, 2),
                            "T": round(T, 2), "C": round(C, 2)}
                )
                edges_added += 1
    
    logger.info(f"Added {edges_added} contradiction edges")

def detect_communities(self) -> List[List[str]]:
    """Detect communities using Louvain method."""
    if not NETWORKX_ADVANCED or self.graph.number_of_edges() == 0:
        return []
    
    try:
        ugraph = self.graph.to_undirected()
        ugraph.remove_edges_from(nx.selfloop_edges(ugraph))
        comms = community.louvain_communities(ugraph, weight='weight')
        return [list(c) for c in comms]
    except Exception as e:
        logger.error(f"Community detection failed: {e}")
        return []

def calculate_centrality(self) -> Dict[str, Dict[str, float]]:
    """Calculate degree and betweenness centrality."""
    if not NETWORKX_ADVANCED or self.graph.number_of_nodes() == 0:
        return {}
    
    try:
        deg_cent = degree_centrality(self.graph)
        bet_cent = betweenness_centrality(self.graph, weight='weight')
        
        return {node: {"degree": deg_cent.get(node, 0.0),
                      "betweenness": bet_cent.get(node, 0.0)}
               for node in self.graph.nodes()}
    except Exception as e:
        logger.error(f"Centrality calculation failed: {e}")
        return {}

def visualize_interactive(self, output_path: Optional[Path] = None):
    """Generate interactive Plotly visualization."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available")
        return
    
    if self.graph.number_of_nodes() == 0:
        return
    
    pos = nx.spring_layout(self.graph, k=2, iterations=50)
    
    # Edge trace
    edge_x, edge_y = [], []
    for edge in self.graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scattergl(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Node trace
    node_x = [pos[node][0] for node in self.graph.nodes()]
    node_y = [pos[node][1] for node in self.graph.nodes()]
    
    node_trace = go.Scattergl(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='lightblue',
            line_width=2))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Tessrax Contradiction Graph',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=40),
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False)))
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_html(fig, str(output_path))
        logger.info(f"Interactive visualization saved: {output_path}")
    else:
        fig.show()

def summary(self, verbose: bool = True):
    """Print comprehensive summary."""
    if not verbose:
        return
    
    print(f"\n{'='*60}")
    print(f"CONFLICT GRAPH SUMMARY")
    print(f"{'='*60}")
    print(f"Nodes: {self.graph.number_of_nodes()}")
    print(f"Edges: {self.graph.number_of_edges()}")
    print(f"Threshold: {self.threshold}")
    
    if self.graph.number_of_edges() > 0:
        print(f"\n--- Top Contradictions ---")
        for u, v, d in list(self.graph.edges(data=True))[:3]:
            print(f"\nScore: {d['weight']:.3f}")
            print(f"  A: {u[:80]}...")
            print(f"  B: {v[:80]}...")
            print(f"  Metrics: {d['metrics']}")

def export_bundle(self) -> Dict[str, Any]:
    """Export complete graph bundle."""
    bundle = {
        "edges": [
            {"s_i": u, "s_j": v, "score": d["weight"], 
             "similarity": d["sim"], "breakdown": d["metrics"]}
            for u, v, d in self.graph.edges(data=True)
        ],
        "stats": {
            "total_statements": self.graph.number_of_nodes(),
            "total_contradictions": self.graph.number_of_edges()
        },
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    
    bundle_json = json.dumps(bundle, sort_keys=True)
    bundle["hash"] = "sha256:" + hashlib.sha256(bundle_json.encode()).hexdigest()
    
    return {"contradiction_bundle": bundle}
```

# ============================================================

# DEMO

# ============================================================

if **name** == “**main**”:
print(“Tessrax Enhanced Conflict Graph - Demo”)
print(”=”*60)

```
sample_texts = [
    {"text": "In 2020, Acme Corp pledged to cut CO2 emissions 50% by 2030.",
     "source": "Press Release"},
    {"text": "In 2024, Acme Corp reported emissions down only 5%.",
     "source": "Annual Report"},
    {"text": "Acme Corp achieved record revenue growth of 40% in 2024.",
     "source": "Investor Call"},
    {"text": "Acme Corp will eliminate all carbon emissions by 2035.",
     "source": "CEO Interview"},
    {"text": "Acme Corp has not reduced carbon emissions significantly.",
     "source": "Audit Report"}
]

cg = ConflictGraph()
cg.add_statements(sample_texts)
cg.compute_edges(verbose=True)
cg.summary()

print("\n✓ Demo complete")
```