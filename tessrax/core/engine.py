#!/usr/bin/env python3
"""
Tessrax Core Engine — Unified Contradiction + Graph Module
----------------------------------------------------------
Now includes a simple Quick-Start entry point so a new user can
run one end-to-end contradiction metabolism demo with a single command.

Usage:
    python -m core.engine
"""

import hashlib, json, os
from collections import defaultdict
from typing import List, Dict, Any
import networkx as nx

from domains.housing.housing_contradiction_detector import detect_contradictions
from core.contradiction_engine import run_contradiction_cycle
from core.governance_kernel import route

# ----------------------------------------------------------------------
# Quick-Start
# ----------------------------------------------------------------------

def quickstart():
    """Run a minimal demonstration of Tessrax."""
    print("\n🧭  Tessrax Quick-Start Demo")
    demo_claims = [
        {"agent": "GPT", "claim": "A", "type": "normative"},
        {"agent": "Claude", "claim": "B", "type": "normative"},
        {"agent": "Gemini", "claim": "A", "type": "normative"},
    ]
    G, stability = run_contradiction_cycle(demo_claims)
    event = route(G, stability)
    print(json.dumps(event.__dict__, indent=2))
    print("\n✅  Demo complete.  Ledger updated at data/governance_ledger.jsonl\n")


if __name__ == "__main__":
    # Detect environment variable QUICKSTART=0 to skip auto-demo (for CI)
    if os.getenv("QUICKSTART", "1") != "0":
        quickstart()