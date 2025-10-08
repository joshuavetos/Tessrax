"""
Tessrax Scaffolding Engine v2.0
Integrated with Governance Kernel

Records conversational design sessions, tracks prompt→response deltas,
and exports structured audit trails that become part of the Tessrax ledger.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from governance_kernel import GovernanceKernel

# ============================================================
# Paths and Setup
# ============================================================

LOG_PATH = Path("data/scaffolding_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize Governance Kernel
kernel = GovernanceKernel()

# ============================================================
# Utility Functions
# ============================================================

def _sha256(obj):
    """Compute SHA-256 hash for any JSON-serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# ============================================================
# Core Recording Functions
# ============================================================

def record_interaction(prompt: str, response: str, tags=None, file_changed=None):
    """
    Append a design-session record.
    tags: list of keywords (e.g. ["governance","forks"])
    file_changed: optional path of the file this session produced
    """
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt.strip(),
        "response": response.strip(),
        "tags": tags or [],
        "file_changed": file_changed,
    }

    # Compute cryptographic hash
    rec["hash"] = "sha256:" + _sha256(rec)

    # Append to scaffolding log
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")

    # Also append as governance event
    kernel.append_event({
        "event": "DESIGN_DECISION_RECORDED",
        "timestamp": rec["timestamp"],
        "file_changed": file_changed,
        "tags": tags or [],
        "decision_hash": rec["hash"],
    })

    return rec["hash"]

# ============================================================
# Summarization
# ============================================================

def summarize_session():
    """
    Summarize scaffolding activity:
    - Total records
    - Distinct tags
    - Last file changed
    """
    if not LOG_PATH.exists():
        return {"records": 0, "tags": [], "last_file": None}

    tags = set()
    last_file = None
    with open(LOG_PATH) as f:
        lines = f.readlines()

    for line in lines:
        try:
            rec = json.loads(line)
            tags.update(rec.get("tags", []))
            if rec.get("file_changed"):
                last_file = rec["file_changed"]
        except Exception:
            pass

    return {
        "records": len(lines),
        "tags": sorted(tags),
        "last_file": last_file,
    }

# ============================================================
# Policy Enforcement (through Kernel Subscriber)
# ============================================================

def enforce_policies(event):
    """
    Example governance policy: prevent mismatched tags and files.
    Enforces consistency between domain tag and affected file.
    """
    if event["event"] != "DESIGN_DECISION_RECORDED":
        return

    tags = event.get("tags", [])
    file_changed = event.get("file_changed", "")

    # Example rule: "fork" tag must only modify fork-related files
    if "fork" in tags and "fork_reconciliation" not in file_changed:
        violation = {
            "event": "POLICY_VIOLATION",
            "policy": "Fork changes must stay within fork_reconciliation_engine.py",
            "file_changed": file_changed,
            "tags": tags,
            "timestamp": datetime.utcnow().isoformat(),
        }
        kernel.append_event(violation)
        print("⚠️ Policy violation recorded:", violation["policy"])

# Register the policy subscriber
kernel.register_subscriber(enforce_policies)

# ============================================================
# Demo Harness
# ============================================================

if __name__ == "__main__":
    print("Running Tessrax Scaffolding Engine v2.0\n")

    h = record_interaction(
        prompt="Integrate scaffolding engine with governance kernel",
        response="Full rewrite with auto-logging and policy enforcement.",
        tags=["governance", "scaffolding"],
        file_changed="scaffolding_engine.py",
    )

    summary = summarize_session()
    print("Recorded design decision:", h)
    print("Session summary:", json.dumps(summary, indent=2))