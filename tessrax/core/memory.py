"""
memory.py — Tessrax Memory v2.1

Contradiction-aware, provenance-tracking memory module.
Supports thread-safe updates and optional JSONL-based persistence.

Implements: IMemory (see interfaces.py)
"""

import json
import threading
from typing import Any, Dict, List, Optional


class Memory:
    """
    In-memory key-value store with:
    - Contradiction detection (flag if value differs for same key)
    - Provenance tracking (list of sources)
    - Thread safety
    - Optional append-only persistence (JSONL)

    Each entry structure:
    {
        "value": Any,
        "provenance": [str],
        "contradiction": bool
    }
    """

    def __init__(self, persist_file: Optional[str] = None):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.persist_file = persist_file
        self.lock = threading.Lock()

        if self.persist_file:
            self._load_from_file()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def add(self, key: str, value: Any, provenance: str) -> None:
        """
        Add or update a key-value pair with provenance.
        - If key is new → add entry
        - If key exists with same value → append provenance only
        - If key exists with different value → flag contradiction
        """
        with self.lock:
            if key not in self.store:
                self.store[key] = {
                    "value": value,
                    "provenance": [provenance],
                    "contradiction": False
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
        """Return a copy of the memory entry for a key, or empty dict if not found."""
        with self.lock:
            return self.store.get(key, {}).copy()

    def export(self) -> str:
        """Return full memory as a JSON-formatted string."""
        with self.lock:
            return json.dumps(self.store, indent=2)

    def all_keys(self) -> List[str]:
        """Return a list of all keys currently stored in memory."""
        with self.lock:
            return list(self.store.keys())

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def _persist_entry(self, key: str) -> None:
        """Append a single key entry to the persistence file (JSONL)."""
        entry = {"key": key, "entry": self.store[key]}
        try:
            with open(self.persist_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"[Warning] Failed to persist memory entry '{key}': {e}")

    def _load_from_file(self) -> None:
        """Load existing memory entries from JSONL file (if it exists)."""
        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        self.store[obj["key"]] = obj["entry"]
        except FileNotFoundError:
            pass  # OK: file will be created on first write
        except Exception as e:
            print(f"[Error] Failed to load memory from file: {e}")


# ------------------------------------------------------------
# Example Usage (Manual Test)
# ------------------------------------------------------------

if __name__ == "__main__":
    mem = Memory("memory.jsonl")

    mem.add("status", "active", "sensor-A")
    mem.add("status", "active", "sensor-B")
    mem.add("status", "inactive", "sensor-C")

    print("[Memory Export]")
    print(mem.export())