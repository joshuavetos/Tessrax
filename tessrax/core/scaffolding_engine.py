"""
Tessrax Scaffolding Engine
Records conversational design sessions, tracks prompt→response deltas,
and exports structured audit trails that become part of the Tessrax ledger.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json
from datetime import datetime
import hashlib
from pathlib import Path

LOG_PATH = Path("data/scaffolding_log.jsonl")

def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def record_interaction(prompt:str, response:str, tags=None, file_changed=None):
    """
    Append a design-session record.
    tags: list of short keywords (e.g. ["governance","forks"])
    file_changed: optional file path this conversation produced
    """
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt.strip(),
        "response": response.strip(),
        "tags": tags or [],
        "file_changed": file_changed,
    }
    rec["hash"] = "sha256:" + _sha256(rec)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec["hash"]

def summarize_session():
    """Return quick stats on how many interactions and distinct tags exist."""
    if not LOG_PATH.exists():
        return {"records":0,"tags":[]}
    tags = set()
    with open(LOG_PATH) as f:
        lines = f.readlines()
    for line in lines:
        try:
            rec = json.loads(line)
            tags.update(rec.get("tags",[]))
        except:
            pass
    return {"records":len(lines),"tags":sorted(tags)}

if __name__ == "__main__":
    h = record_interaction(
        "Example prompt about fork reconciliation",
        "Response: integrated relativistic merge pattern.",
        tags=["example","demo"],
        file_changed="fork_reconciliation_engine.py"
    )
    print("Recorded interaction:", h)
    print("Summary:", summarize_session())