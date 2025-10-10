"""
Ledger Core — unified append-only audit log and chain verification.
Combines contradiction_ledger.py and ledger logic from Ledger.txt.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path


def hash_entry(entry: dict, prev_hash: str = "0") -> str:
    """Compute SHA-256 hash for a ledger entry + previous hash."""
    raw = json.dumps(entry, sort_keys=True).encode() + prev_hash.encode()
    return hashlib.sha256(raw).hexdigest()


def append_entry(entry: dict, ledger_path="data/ledger.jsonl"):
    """Append a new entry to the ledger with hash chaining."""
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    prev_hash = "0"
    try:
        with open(ledger_path, "r") as f:
            for line in f:
                data = json.loads(line)
                prev_hash = data.get("hash", prev_hash)
    except FileNotFoundError:
        pass

    entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    entry["prev_hash"] = prev_hash
    entry["hash"] = hash_entry(entry, prev_hash)

    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["hash"]


def verify_chain(ledger_path="data/ledger.jsonl"):
    """Verify integrity of the ledger hash chain."""
    prev_hash = "0"
    with open(ledger_path, "r") as f:
        for i, line in enumerate(f, start=1):
            entry = json.loads(line)
            calc = hash_entry(entry, prev_hash)
            if calc != entry["hash"]:
                return False, i
            prev_hash = entry["hash"]
    return True, None
