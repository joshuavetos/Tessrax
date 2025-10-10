"""
Tessrax Governance Ledger
--------------------------

Append-only, hash-chained ledger for contradiction and governance events.
Provides lightweight integrity verification and audit retrieval.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional


class GovernanceLedger:
    def __init__(self, path: str = "data/governance_ledger.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            self.path.write_text("")

    def _get_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        with open(self.path, "r", encoding="utf-8") as f:
            try:
                last = json.loads(f.readlines()[-1])
                return last.get("hash", "0" * 64)
            except Exception:
                return "0" * 64

    def add_event(self, data: Dict[str, any]) -> Dict[str, any]:
        """Append new governance event and return record with computed hash."""
        prev_hash = self._get_last_hash()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prev_hash": prev_hash,
            "event": data,
        }
        record_str = json.dumps(record, sort_keys=True)
        record["hash"] = hashlib.sha256(record_str.encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def read_all(self) -> List[Dict[str, any]]:
        """Return all ledger entries."""
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify_chain(self) -> bool:
        """Verify hash chaining integrity."""
        lines = self.read_all()
        prev_hash = "0" * 64
        for entry in lines:
            if entry.get("prev_hash") != prev_hash:
                return False
            recomputed = hashlib.sha256(
                json.dumps(
                    {"timestamp": entry["timestamp"], "prev_hash": entry["prev_hash"], "event": entry["event"]},
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if recomputed != entry["hash"]:
                return False
            prev_hash = entry["hash"]
        return True