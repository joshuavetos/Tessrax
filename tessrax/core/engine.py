#!/usr/bin/env python3
"""
Tessrax Core Engine — Unified Contradiction Metabolism Runner (v4)
------------------------------------------------------------------

Provides a single command-line entry point to run, inspect, and extend
the Tessrax governance framework.

Major Features
--------------
• Single entry point for demos, multi-domain runs, and ledger inspection
• Colab-safe dynamic path resolution
• Graceful degradation when optional domains are missing
• Multi-domain execution with --domain all
• Optional output formats (--format summary/json)
• Interactive REPL mode for manual contradiction testing
• Ledger inspection and verification utilities
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import networkx as nx

# ----------------------------------------------------------------------
# Path Setup (Colab-safe)
# ----------------------------------------------------------------------
PROJECT_ROOT = "/content/Tessrax-main/Tessrax-main/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------------------------------------------------
# Core Imports
# ----------------------------------------------------------------------
from tessrax.core.contradiction_engine import (
    detect_contradictions,
    score_stability,
    log_to_ledger,
)
from tessrax.core.governance_kernel import route

# ----------------------------------------------------------------------
# Optional Domain Imports
# ----------------------------------------------------------------------
def try_import_domain(domain_name: str, path: str):
    """Attempt to import a domain detector gracefully."""
    try:
        module = __import__(path, fromlist=["detect_contradictions"])
        return getattr(module, "detect_contradictions"), True
    except Exception:
        return lambda: {"error": f"{domain_name} detector unavailable"}, False


detect_housing_contradictions, HOUSING_AVAILABLE = try_import_domain(
    "housing", "domains.housing.housing_contradiction_detector"
)
detect_memory_conflicts, MEMORY_AVAILABLE = try_import_domain(
    "ai_memory", "domains.ai_memory.memory_contradiction_detector"
)
detect_attention_conflicts, ATTENTION_AVAILABLE = try_import_domain(
    "attention", "domains.attention.attention_contradiction_detector"
)
detect_governance_conflicts, GOVERNANCE_AVAILABLE = try_import_domain(
    "governance", "domains.democratic_governance.governance_contradiction_detector"
)
detect_climate_conflicts, CLIMATE_AVAILABLE = try_import_domain(
    "climate", "domains.climate_policy.climate_contradiction_detector"
)

AVAILABLE_DOMAINS = {
    "housing": (detect_housing_contradictions, HOUSING_AVAILABLE),
    "ai_memory": (detect_memory_conflicts, MEMORY_AVAILABLE),
    "attention": (detect_attention_conflicts, ATTENTION_AVAILABLE),
    "governance": (detect_governance_conflicts, GOVERNANCE_AVAILABLE),
    "climate": (detect_climate_conflicts, CLIMATE_AVAILABLE),
}

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def pretty_print(header: str, obj: Dict[str, Any], color=True):
    """Human-friendly colored JSON output with fallback."""
    print(f"\n\033[96m{header}\033[0m" if color else f"\n{header}")
    try:
        from pygments import highlight
        from pygments.lexers import JsonLexer
        from pygments.formatters import TerminalFormatter

        json_str = json.dumps(obj, indent=2)
        if color:
            print(highlight(json_str, JsonLexer(), TerminalFormatter()))
        else:
            print(json_str)
    except Exception:
        print(json.dumps(obj, indent=2))


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file if provided."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_ledger(path: str = "data/governance_ledger.jsonl", count: int = 5):
    """Print the last few ledger entries."""
    if not Path(path).exists():
        print("Ledger file not found.")
        return
    lines = Path(path).read_text().splitlines()
    recent = lines[-count:] if len(lines) >= count else lines
    print(f"\n📜 Showing {len(recent)} most recent ledger entries:\n")
    for line in recent:
        print(json.dumps(json.loads(line), indent=2))


def verify_ledger_integrity(path: str = "data/governance_ledger.jsonl"):
    """Check hash chaining integrity."""
    if not Path(path).exists():
        print("Ledger not found.")
        return
    lines = Path(path).read_text().splitlines()
    prev_hash = "0" * 64
    for i, line in enumerate(lines):
        entry = json.loads(line)
        expected_prev = entry.get("prev_hash", "")
        if expected_prev != prev_hash:
            print(f"❌ Hash mismatch at line {i}")
            return
        prev_hash = entry.get("hash", "")
    print("✅ Ledger hash chain verified.")


# ----------------------------------------------------------------------
# Main Execution Flow
# ----------------------------------------------------------------------
def run_core_demo():
    """Run the default contradiction metabolism demo."""
    print("\n🧭  Tessrax Quick-Start Demo\n")
    demo_claims = [
        {"agent": "Alice", "claim": "The door is open.", "type": "fact"},
        {"agent": "Bob", "claim": "The door is not open.", "type": "fact"},
        {"agent": "Charlie", "claim": "The cat is on the mat.", "type": "fact"},
        {"agent": "HousingAgent", "claim": "Property values are increasing.", "type": "fact"},
    ]
    print("🔍 Detecting contradictions...")
    G = detect_contradictions(demo_claims)
    stability = score_stability(G)
    print(f"   Stability index: {stability:.3f}")

    print("\n⚖️  Routing to governance kernel...")
    event = route(G, stability)
    event_dict = (
        event.__dict__ if hasattr(event, "__dict__")
        else event if isinstance(event, dict)
        else {"event": str(event)}
    )
    pretty_print("Governance Event", event_dict)
    log_to_ledger(G, stability, "data/governance_ledger.jsonl")
    print("\n✅ Demo complete. Ledger updated at data/governance_ledger.jsonl\n")


def run_domain(name: str):
    """Run a specific domain detector if available."""
    detector, available = AVAILABLE_DOMAINS.get(name, (None, False))
    if not available or not detector:
        print(f"⚠️ Domain '{name}' is unavailable.")
        return
    print(f"\n🔬 Running domain detector: {name}")
    result = detector()
    pretty_print(f"{name.title()} Domain Output", result)


def run_all_domains():
    """Run all available domain detectors."""
    for name, (detector, available) in AVAILABLE_DOMAINS.items():
        if available:
            run_domain(name)
        else:
            print(f"⚠️ {name.title()} domain missing or unavailable.")


def interactive_mode():
    """Simple REPL for manual contradiction testing."""
    print("🧮 Tessrax Interactive Mode — type 'exit' to quit.")
    while True:
        text = input("\nEnter two comma-separated claims: ")
        if text.strip().lower() == "exit":
            break
        try:
            a, b = [t.strip() for t in text.split(",", 1)]
            claims = [{"agent": "A", "claim": a}, {"agent": "B", "claim": b}]
            G = detect_contradictions(claims)
            stability = score_stability(G)
            pretty_print("Result", {"stability_index": stability})
        except Exception as e:
            print(f"Error: {e}")


# ----------------------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Tessrax engine demos and tools.")
    parser.add_argument("--domain", type=str, help="Specify a domain (or 'all').")
    parser.add_argument("--format", type=str, default="summary",
                        choices=["summary", "json"], help="Output format.")
    parser.add_argument("--inspect-ledger", action="store_true", help="Show recent ledger entries.")
    parser.add_argument("--verify-ledger", action="store_true", help="Check hash chain integrity.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL.")
    parser.add_argument("--config", type=str, help="Path to custom config JSON.")
    args = parser.parse_args()

    if args.inspect_ledger:
        inspect_ledger()
        return
    if args.verify_ledger:
        verify_ledger_integrity()
        return
    if args.interactive:
        interactive_mode()
        return

    config = load_config(args.config) if args.config else {}
    selected_domain = args.domain or config.get("domain")

    if selected_domain == "all":
        run_all_domains()
    elif selected_domain:
        run_domain(selected_domain)
    else:
        run_core_demo()


if __name__ == "__main__":
    main()