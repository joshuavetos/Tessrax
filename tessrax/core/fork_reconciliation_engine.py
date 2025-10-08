"""
Tessrax Fork Reconciliation Engine
Implements fork-tolerant ledger merging, relativistic trust windows,
and unobserved-compromise handling.

Upgraded version with hash chaining, timestamp validation, and persistent logging.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import hashlib
import json
from datetime import datetime
from dateutil.parser import parse

# ============================================================
# Utility Functions
# ============================================================

def sha256(data):
    """Compute a stable SHA-256 hash for any JSON-serializable object."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def safe_parse_time(ts):
    """Parse timestamps safely; fallback to UTC now if invalid."""
    try:
        return parse(ts)
    except Exception:
        return datetime.utcnow()

def hash_manifest_union(manifests):
    """Compute hash representing merged manifest state."""
    concat = "".join(sorted(manifests.keys()))
    return "sha256:" + hashlib.sha256(concat.encode()).hexdigest()

# ============================================================
# Ledger Persistence + Hash Chaining
# ============================================================

def append_with_chain(ledger, entry):
    """Append entry to ledger with hash chaining for immutability."""
    prev = ledger[-1]["hash"] if ledger else None
    entry["prev_hash"] = prev
    entry["hash"] = "sha256:" + sha256(entry)
    ledger.append(entry)
    return ledger

def commit_ledger(ledger, path="data/ledger.jsonl"):
    """Persist ledger entries to disk in append-only mode."""
    with open(path, "a") as f:
        for rec in ledger:
            f.write(json.dumps(rec) + "\n")
    print(f"Ledger committed to {path} ({len(ledger)} records)")

# ============================================================
# Relativistic Trust Window
# ============================================================

def trust_window_global(key_entry):
    """
    Compute trust window as (activation, max(revocation_times)).
    key_entry = {
        "key": "ed25519:2025Q4-001",
        "activation_time": datetime,
        "revocation_times": {"A": datetime, "B": datetime}
    }
    """
    start = key_entry["activation_time"]
    end = max(key_entry["revocation_times"].values())
    return (start, end)

# ============================================================
# Fork Reconciliation
# ============================================================

def reconcile_branches(branch_a, branch_b, manifests):
    """
    Merge two ledger branches without losing data.
    Returns merged receipts + merged manifest reference.
    """
    merged = []
    ids_a = {r["receipt_id"] for r in branch_a}
    ids_b = {r["receipt_id"] for r in branch_b}
    shared = ids_a & ids_b

    # Add shared receipts
    for r in branch_a:
        if r["receipt_id"] in shared:
            merged.append(r)

    # Add branch A deltas
    for r in branch_a:
        if r["receipt_id"] not in ids_b:
            r["conflicted"] = True
            r["reason"] = "branch_A_only"
            merged.append(r)

    # Add branch B deltas
    for r in branch_b:
        if r["receipt_id"] not in ids_a:
            r["conflicted"] = True
            r["reason"] = "branch_B_only"
            merged.append(r)

    merged_manifest = {
        "merged_manifest": hash_manifest_union(manifests),
        "history": list(manifests.keys())
    }

    return merged, merged_manifest

# ============================================================
# Unobserved Compromise Handler
# ============================================================

def handle_unobserved_compromise(branch, key_id, global_revocation_time):
    """
    Tag receipts signed by keys that were later revoked, but the branch never observed it.
    """
    for r in branch:
        ts = safe_parse_time(r.get("timestamp"))
        if (
            r.get("manifest_ref") == key_id
            and ts < global_revocation_time
        ):
            r["conflicted"] = True
            r["reason"] = "unobserved_compromise"
            r["local_validity"] = {"branch_local": True, "global_view": False}
    return branch

# ============================================================
# Governance Merge Commit
# ============================================================

def merge_and_commit(branch_a, branch_b, manifests, ledger, path="data/ledger.jsonl"):
    """
    Combine two ledger branches, reconcile manifests, append merge record, and persist.
    """
    merged_receipts, merged_manifest = reconcile_branches(branch_a, branch_b, manifests)
    merge_event = {
        "event": "BRANCH_MERGE_EVENT",
        "merged_manifest": merged_manifest,
        "timestamp": datetime.utcnow().isoformat(),
        "receipts_merged": len(merged_receipts)
    }
    append_with_chain(ledger, merge_event)
    for r in merged_receipts:
        append_with_chain(ledger, r)
    commit_ledger(ledger, path)
    return ledger

# ============================================================
# Demo / Test Harness
# ============================================================

if __name__ == "__main__":
    manifests = {
        "sha256:manif_2025Q4-001": {"keys": ["key_001"]},
        "sha256:manif_2025Q4-002": {"keys": ["key_002"]},
    }

    branch_a = [
        {"receipt_id": "r1", "event": "A1", "timestamp": "2025-10-07T00:00:00", "manifest_ref": "sha256:manif_2025Q4-001"},
        {"receipt_id": "r2", "event": "A2", "timestamp": "2025-10-07T01:00:00", "manifest_ref": "sha256:manif_2025Q4-001"},
    ]
    branch_b = [
        {"receipt_id": "r1", "event": "A1", "timestamp": "2025-10-07T00:00:00", "manifest_ref": "sha256:manif_2025Q4-001"},
        {"receipt_id": "r3", "event": "B1", "timestamp": "2025-10-07T02:00:00", "manifest_ref": "sha256:manif_2025Q4-002"},
    ]

    ledger = []
    ledger = merge_and_commit(branch_a, branch_b, manifests, ledger)
    print(json.dumps(ledger, indent=2))