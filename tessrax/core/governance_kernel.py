#!/usr/bin/env python3
"""
Tessrax Governance Kernel — Unified Orchestration + Policy Layer
----------------------------------------------------------------
Merged modules:
  • policy_rules.py
  • orchestrator.py
  • resource_guard.py

Purpose:
  Central coordination point for the Tessrax system.
  Routes contradiction results into appropriate governance lanes,
  enforces policy constraints, and guards key runtime resources.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tessrax.core.engine import detect_contradictions, export_governance_case
from tessrax.core.ledger import append_entry, summarize_ledger, verify_chain


# ----------------------------------------------------------------------
# Core Constants
# ----------------------------------------------------------------------

LANE_DESCRIPTIONS = {
    "Autonomic": "Low-risk consensus — system may self-correct or auto-adopt.",
    "Deliberative": "Moderate conflict — requires human or quorum review.",
    "Constitutional": "High conflict — challenges base rules or principles.",
    "Behavioral Audit": "Critical — indicates drift, manipulation, or malice.",
}

POLICY_REGISTRY = {
    "max_conflict_ratio": 0.25,
    "auto_commit_threshold": 0.8,
    "require_quorum_for": ["Deliberative", "Constitutional", "Behavioral Audit"],
    "guarded_resources": ["ledger.jsonl", "tessrax/core/memory.py"],
}

# ----------------------------------------------------------------------
# Governance Orchestration
# ----------------------------------------------------------------------

def route(agent_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Full governance pipeline: detect contradictions → classify → log to ledger.
    """
    graph = detect_contradictions(agent_claims)
    case = export_governance_case(graph)
    lane = case["lane"]
    stability = case["stability_index"]

    print(f"\n[GovernanceKernel] Case {case['case_id']} routed to lane: {lane} (stability={stability})")

    # Policy enforcement
    if lane == "Autonomic" and stability >= POLICY_REGISTRY["auto_commit_threshold"]:
        action = "auto-commit"
    elif lane in POLICY_REGISTRY["require_quorum_for"]:
        action = "human-review"
    else:
        action = "queue"

    entry = {
        "case_id": case["case_id"],
        "lane": lane,
        "stability_index": stability,
        "agents": case["agents"],
        "action": action,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reason": case["reason"],
    }

    append_entry(entry)
    return entry


# ----------------------------------------------------------------------
# Policy Evaluation
# ----------------------------------------------------------------------

def evaluate_policy_conflicts(policies: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check a given policy set for contradictions or violations of core principles.
    """
    violations = []

    if policies.get("auto_commit_threshold", 1.0) > 1.0:
        violations.append("auto_commit_threshold cannot exceed 1.0")

    if "max_conflict_ratio" not in policies or policies["max_conflict_ratio"] <= 0:
        violations.append("max_conflict_ratio must be positive and defined")

    # Reserved keyword protection
    for key in policies:
        if key.startswith("_") or key.lower() in ["delete", "drop", "overwrite"]:
            violations.append(f"Invalid or dangerous policy key: {key}")

    return (len(violations) == 0, violations)


# ----------------------------------------------------------------------
# Resource Guard (formerly resource_guard.py)
# ----------------------------------------------------------------------

def guard_resources() -> List[str]:
    """
    Ensure critical resources exist and are unmodified.
    Returns list of warnings (if any).
    """
    warnings = []

    # Verify ledger chain integrity
    ok, line = verify_chain()
    if not ok:
        warnings.append(f"Ledger chain broken at line {line}")

    # Verify guarded files exist
    for res in POLICY_REGISTRY["guarded_resources"]:
        path = Path(res)
        if not path.exists():
            warnings.append(f"Missing guarded resource: {res}")
        else:
            size = path.stat().st_size
            if size < 10:
                warnings.append(f"Resource {res} appears empty or corrupted.")

    return warnings


# ----------------------------------------------------------------------
# Quorum Simulation
# ----------------------------------------------------------------------

def simulate_quorum_review(case_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a quorum review decision for higher-risk lanes.
    """
    lane = case_entry["lane"]
    decision = "approved" if lane == "Deliberative" else "escalated"
    confidence = 0.7 if lane == "Deliberative" else 0.4

    quorum_result = {
        "case_id": case_entry["case_id"],
        "lane": lane,
        "decision": decision,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    append_entry(quorum_result)
    return quorum_result


# ----------------------------------------------------------------------
# System Status
# ----------------------------------------------------------------------

def system_health_summary() -> Dict[str, Any]:
    """
    Compile a full snapshot of governance system health.
    """
    warnings = guard_resources()
    ledger_status = summarize_ledger()

    return {
        "system": "Tessrax Governance Kernel",
        "warnings": warnings,
        "ledger_summary": ledger_status,
        "active_policies": POLICY_REGISTRY,
    }


# ----------------------------------------------------------------------
# Demo Entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Tessrax Governance Kernel Demo ---")

    # Example claims for demo
    claims = [
        {"agent": "GPT", "claim": "A"},
        {"agent": "Claude", "claim": "B"},
        {"agent": "Gemini", "claim": "A"},
    ]

    # Run through kernel
    case = route(claims)
    print(json.dumps(case, indent=2))

    # Simulate quorum
    if case["lane"] != "Autonomic":
        quorum = simulate_quorum_review(case)
        print("\n[Quorum Result]")
        print(json.dumps(quorum, indent=2))

    # Show health
    print("\n[System Health Summary]")
    print(json.dumps(system_health_summary(), indent=2))
