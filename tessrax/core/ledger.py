#!/usr/bin/env python3
"""
Tessrax Ledger — Immutable Append-Only Governance Ledger
---------------------------------------------------------
Merged modules:
  • contradiction_ledger.py
  • contradiction_ledger_v4.json
  • ledger_security_upgrade.py

Purpose:
  Maintain a tamper-evident, cryptographically chained record of
  governance events, contradiction analyses, and agent outcomes.
"""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

LEDGER_PATH = Path("data/ledger.jsonl")


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_ledger_dir():
    Path("data").mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Ledger Core
# ----------------------------------------------------------------------

def _get_last_entry() -> Tuple[str, str]:
    """Return (prev_hash, last_line) of the ledger if exists."""
    if not LEDGER_PATH.exists():
        return ("0" * 64, None)

    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return ("0" * 64, None)
        last_entry = json.loads(lines[-1])
        return (last_entry["hash"], lines[-1].strip())


def append_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append an entry to the ledger with cryptographic chaining.

    Args:
        entry: dict of governance or contradiction data.

    Returns:
        The finalized ledger entry with hash + prev_hash fields.
    """
    _ensure_ledger_dir()
    prev_hash, _ = _get_last_entry()

    # Add standard ledger metadata
    entry["timestamp"] = _timestamp()
    entry["prev_hash"] = prev_hash
    entry_str = json.dumps(entry, sort_keys=True)
    entry_hash = _sha256(entry_str)

    entry["hash"] = entry_hash

    # Append to ledger
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


# ----------------------------------------------------------------------
# Verification + Audit
# ----------------------------------------------------------------------

def verify_chain(filepath: Path = LEDGER_PATH) -> Tuple[bool, int]:
    """
    Verify integrity of the ledger chain.
    Returns (True, -1) if valid; (False, line_number) if corrupted.
    """
    if not filepath.exists():
        return True, -1

    with open(filepath, "r", encoding="utf-8") as f:
        prev_hash = "0" * 64
        for i, line in enumerate(f, start=1):
            entry = json.loads(line)
            data_copy = dict(entry)
            actual_hash = data_copy.pop("hash", None)
            expected_hash = _sha256(json.dumps(data_copy, sort_keys=True))

            if expected_hash != actual_hash or entry.get("prev_hash") != prev_hash:
                return False, i
            prev_hash = actual_hash
    return True, -1


# ----------------------------------------------------------------------
# Query Interface
# ----------------------------------------------------------------------

def load_all_entries() -> List[Dict[str, Any]]:
    """Load the entire ledger into memory."""
    if not LEDGER_PATH.exists():
        return []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_entries_by_agent(agent_name: str) -> List[Dict[str, Any]]:
    """Return all entries mentioning a given agent."""
    entries = load_all_entries()
    return [e for e in entries if agent_name in str(e)]


def summarize_ledger() -> Dict[str, Any]:
    """Produce a quick health summary."""
    valid, break_line = verify_chain()
    entries = load_all_entries()
    return {
        "valid_chain": valid,
        "break_line": break_line if not valid else None,
        "entry_count": len(entries),
        "latest_entry": entries[-1] if entries else None,
    }


# ----------------------------------------------------------------------
# Demo CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Tessrax Ledger Demo ---")

    # Append example entries
    entry = {
        "agents": ["PolicyAI", "AuditBot"],
        "stability_index": 0.82,
        "governance_lane": "Autonomic",
        "note": "Consensus reached on bias testing protocols; auto-adopted."
    }

    new_entry = append_entry(entry)
    print("Appended:", json.dumps(new_entry, indent=2))

    # Verify integrity
    ok, break_line = verify_chain()
    print(f"Ledger integrity: {'OK' if ok else f'Corrupted at line {break_line}'}")

    # Summarize
    summary = summarize_ledger()
    print(json.dumps(summary, indent=2))
