# ledger_core.py  — Unified Governance Ledger (v1.1)

import json, hashlib, os, datetime
from pathlib import Path

LEDGER_PATH = Path("data/governance_ledger.jsonl")

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

def append_event(event: dict, ledger_path: Path = LEDGER_PATH):
    """Append event with hash-chain continuity."""
    os.makedirs(ledger_path.parent, exist_ok=True)
    prev_hash = None
    if ledger_path.exists():
        with ledger_path.open() as f:
            for line in f:
                pass
            if line.strip():
                prev_hash = json.loads(line)["hash"]

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "event": event,
        "prev_hash": prev_hash,
    }
    payload = json.dumps(entry, sort_keys=True)
    entry["hash"] = _sha256(payload)
    with ledger_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["hash"]

def verify_chain(ledger_path: Path = LEDGER_PATH):
    """Return True if ledger hash chain is intact."""
    prev = None
    with ledger_path.open() as f:
        for line in f:
            e = json.loads(line)
            expected = _sha256(json.dumps({k:v for k,v in e.items() if k!="hash"}, sort_keys=True))
            if expected != e["hash"] or e.get("prev_hash") != prev:
                return False
            prev = e["hash"]
    return True
Migration step

In engine_core.py and governance_kernel.py, replace all log_to_ledger(...) or append_event(...) calls with:
from tessrax.core.ledger_core import append_event
append_event(result)