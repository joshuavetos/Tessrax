"""
Tessrax Reconciliation Engine
-----------------------------

Analyzes the governance ledger to identify recurring or resolved contradictions.
"""

import json
from pathlib import Path
from typing import Dict, Any


def reconcile(ledger_path: str = "data/governance_ledger.jsonl") -> Dict[str, Any]:
    """Compute reconciliation summary."""
    path = Path(ledger_path)
    if not path.exists():
        return {"error": "Ledger not found"}

    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = {}
    resolved = 0
    for entry in entries:
        agents = tuple(sorted(entry["event"].get("agents", [])))
        if agents in seen:
            resolved += 1
        seen[agents] = entry
    return {
        "total_entries": len(entries),
        "unique_agent_pairs": len(seen),
        "resolved_conflicts": resolved,
        "unresolved_conflicts": len(seen) - resolved,
    }