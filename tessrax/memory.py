"""
memory.py - Tessrax v2.0
Contradiction-aware, provenance-tracking, thread-safe memory with optional persistent JSONL storage.
"""

import json
import threading
from typing import Any, Dict, Optional, List

class Memory:
    def __init__(self, persist_file: Optional[str] = None):
        """
        Initialize memory.
        :param persist_file: Path to a JSONL file for persistent storage (optional).
        """
        self.store: Dict[str, Dict[str, Any]] = {}
        self.persist_file = persist_file
        self.lock = threading.Lock()
        if self.persist_file:
            self._load_from_file()

    def add(self, key: str, value: Any, provenance: str) -> None:
        """
        Add a key/value with provenance.
        If the same value already exists, no contradiction is set.
        If a different value exists, contradiction is flagged and provenance extended.
        """
        with self.lock:
            if key not in self.store:
                self.store[key] = {
                    "value": value,
                    "provenance": [provenance],
                    "contradiction": False,
                }
            else:
                entry = self.store[key]
                if entry["value"] != value:
                    entry["contradiction"] = True
                if provenance not in entry["provenance"]:
                    entry["provenance"].append(provenance)
            if self.persist_file:
                self._persist_entry(key)

    def get(self, key: str) -> Dict[str, Any]:
        """Get the entry for a key (empty dict if not found)."""
        with self.lock:
            return self.store.get(key, {}).copy()

    def export(self) -> str:
        """Export all memory contents as a pretty-printed JSON string."""
        with self.lock:
            return json.dumps(self.store, indent=2)

    def _load_from_file(self):
        """Load existing entries from the JSONL file (if present)."""
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.store[obj["key"]] = obj["entry"]
        except FileNotFoundError:
            pass  # No persisted memory yet

    def _persist_entry(self, key: str):
        """Persist a single entry (append to JSONL file)."""
        entry = {"key": key, "entry": self.store[key]}
        with open(self.persist_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def all_keys(self) -> List[str]:
        """Return all keys in memory."""
        with self.lock:
            return list(self.store.keys())

if __name__ == "__main__":
    mem = Memory("memory.jsonl")
    mem.add("status", "active", "sensor-A")
    mem.add("status", "active", "sensor-B")
    mem.add("status", "inactive", "sensor-C")
    print(mem.export())