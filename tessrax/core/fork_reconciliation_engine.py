"""
Tessrax Fork Reconciliation Engine
Implements fork-tolerant ledger merging, relativistic trust windows,
and unobserved-compromise handling.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import hashlib
import json
from datetime import datetime

# ============================================================
# Utility Functions
# ============================================================

def sha256(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def hash_manifest_union(manifests):
    """Compute hash representing merged manifest state."""
    concat = "".join(sorted(manifests.keys()))
    return "sha256:" + hashlib.sha256(concat.encode()).hexdigest()

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
        if (
            r.get("manifest_ref") == key_id
            and datetime.fromisoformat(r["timestamp"]) < global_revocation_time
        ):
            r["conflicted"] = True
            r["reason"] = "unobserved_compromise"
            r["local_validity"] = {"branch_local": True, "global_view": False}
    return branch

# ============================================================
# Governance Merge Commit
# ============================================================

def merge_and_commit(branch_a, branch_b, manifests, ledger):
    """
    Combine two ledger branches, reconcile manifests, and append merge record.
    """
    merged_receipts, merged_manifest = reconcile_branches(branch_a, branch_b, manifests)
    merge_event = {
        "event": "BRANCH_MERGE_EVENT",
        "merged_manifest": merged_manifest,
        "timestamp": datetime.utcnow().isoformat(),
        "receipts_merged": len(merged_receipts),
        "hash": "sha256:" + sha256(merged_receipts),
    }
    ledger.append(merge_event)
    for r in merged_receipts:
        ledger.append(r)
    return ledger

# ============================================================
# Demo / Test Harness
# ============================================================

if __name__ == "__main__":
    # Example manifests and branches
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