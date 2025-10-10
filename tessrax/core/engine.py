#!/usr/bin/env python3
"""
Tessrax Core Engine — Unified Contradiction Metabolism Runner
-------------------------------------------------------------

Acts as the single entry point for the Tessrax system.

Features:
    • Dynamically configures import paths (for Colab or local use)
    • Runs the full contradiction metabolism demo end-to-end
    • Optionally executes domain-specific detectors (e.g., housing)
    • Writes results to `data/governance_ledger.jsonl`

Usage:
    python core/engine.py             # Run default demo
    python core/engine.py --domain housing
    QUICKSTART=0 python core/engine.py    # Skip demo (for imports / tests)
"""

import os
import sys
import json
import argparse
import networkx as nx
from pathlib import Path
from typing import Dict, Any

# ----------------------------------------------------------------------
# Dynamic import path setup (Colab safe)
# ----------------------------------------------------------------------
PROJECT_ROOT = "/content/Tessrax-main/Tessrax-main/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------------------
# Core imports
# ----------------------------------------------------------------------
from tessrax.core.contradiction_engine import (
    detect_contradictions,
    score_stability,
    log_to_ledger,
)
from tessrax.core.governance_kernel import route

# ----------------------------------------------------------------------
# Optional domain import
# ----------------------------------------------------------------------
try:
    from domains.housing.housing_contradiction_detector import detect_contradictions as detect_housing_contradictions
    HOUSING_AVAILABLE = True
except Exception:
    HOUSING_AVAILABLE = False

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def pretty_print(header: str, obj: Dict[str, Any]):
    """Nicely formatted JSON output with color."""
    try:
        import pygments
        from pygments import highlight
        from pygments.lexers import JsonLexer
        from pygments.formatters import TerminalFormatter

        json_str = json.dumps(obj, indent=2)
        print(f"\033[96m{header}\033[0m")
        print(highlight(json_str, JsonLexer(), TerminalFormatter()))
    except Exception:
        # Fallback without color
        print(f"\n{header}\n{json.dumps(obj, indent=2)}\n")


# ----------------------------------------------------------------------
# Quick-start demo
# ----------------------------------------------------------------------
def quickstart(selected_domain: str = None):
    """Run an end-to-end Tessrax demonstration."""
    print("\n🧭  Tessrax Quick-Start Demo\n")

    demo_claims = [
        {"agent": "Alice", "claim": "The door is open.", "type": "fact"},
        {"agent": "Bob", "claim": "The door is not open.", "type": "fact"},
        {"agent": "Charlie", "claim": "The cat is on the mat.", "type": "fact"},
        {"agent": "HousingAgent", "claim": "Property values are increasing.", "type": "fact"},
    ]

    print("🔍 Running core contradiction detection...")
    graph = detect_contradictions(demo_claims)
    stability = score_stability(graph)
    print(f"   Stability index: {stability:.3f}")

    # Run domain-specific detector if requested and available
    if selected_domain == "housing" and HOUSING_AVAILABLE:
        print("\n🏘️  Running housing domain detector...")
        housing_result = detect_housing_contradictions()
        pretty_print("Housing detector output", housing_result)
    elif selected_domain == "housing":
        print("\n⚠️  Housing domain detector not available.")

    print("\n⚖️  Routing core contradiction report...")
    event = route(graph, stability)
    event_dict = (
        event.__dict__ if hasattr(event, "__dict__") else
        event if isinstance(event, dict) else
        {"event": str(event)}
    )
    pretty_print("Governance Event", event_dict)

    print("\n🪶 Writing to ledger...")
    Path("data").mkdir(exist_ok=True)
    log_to_ledger(graph, stability, "data/governance_ledger.jsonl")
    print("\n✅  Demo complete. Ledger updated at data/governance_ledger.jsonl\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Tessrax Core Engine demo.")
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional domain to run (e.g., 'housing').",
    )
    args = parser.parse_args()

    if os.getenv("QUICKSTART", "1") != "0":
        quickstart(args.domain)
    else:
        print("QUICKSTART=0 set — skipping demo run.")


if __name__ == "__main__":
    main()
