"""
Tessrax Scaffolding Engine v2.1
Logs design sessions and feeds them into Governance Kernel.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json, hashlib
from datetime import datetime
from pathlib import Path
from governance_kernel import GovernanceKernel

LOG_PATH = Path("data/scaffolding_log.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
kernel = GovernanceKernel()

def _sha256(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def record_interaction(prompt, response, tags=None, file_changed=None):
    """Record design conversation and send to governance."""
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt.strip(),
        "response": response.strip(),
        "tags": tags or [],
        "file_changed": file_changed,
    }
    rec["hash"] = "sha256:" + _sha256(rec)

    with open(LOG_PATH, "a") as f: f.write(json.dumps(rec) + "\n")

    # Notify governance
    kernel.append_event({
        "event": "DESIGN_DECISION_RECORDED",
        "file_changed": file_changed,
        "tags": tags or [],
        "decision_hash": rec["hash"],
        "timestamp": rec["timestamp"]
    })

    return rec["hash"]

def summarize_session():
    """Quick summary of session log."""
    if not LOG_PATH.exists(): return {"records": 0, "tags": [], "last_file": None}
    tags, last_file = set(), None
    with open(LOG_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                tags.update(rec.get("tags", []))
                if rec.get("file_changed"): last_file = rec["file_changed"]
            except: pass
    return {"records": sum(1 for _ in open(LOG_PATH)), "tags": sorted(tags), "last_file": last_file}

if __name__ == "__main__":
    print("Running Tessrax Scaffolding Engine v2.1 …")
    h = record_interaction(
        prompt="Integrate dynamic policy rules",
        response="Added policy_rules.py with declarative enforcement.",
        tags=["governance", "scaffolding"],
        file_changed="policy_rules.py",
    )
    print("Recorded:", h)
    print("Summary:", json.dumps(summarize_session(), indent=2))