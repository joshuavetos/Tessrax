"""
memory.py - Tessrax v1.0
Contradiction-aware memory with provenance tracking.
"""

import json
from typing import Any, Dict


class Memory:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def add(self, key: str, value: Any, provenance: str) -> None:
        """
        Add a key/value with provenance.
        If the same value already exists, no contradiction is set.
        If a different value exists, contradiction=True.
        """
        if key not in self.store:
            self.store[key] = {
                "value": value,
                "provenance": provenance,
                "contradiction": False,
            }
        else:
            entry = self.store[key]
            if entry["value"] != value:
                entry["contradiction"] = True
            # provenance can be extended
            entry["provenance"] += f", {provenance}"

    def get(self, key: str) -> Dict[str, Any]:
        return self.store.get(key, {})

    def export(self) -> str:
        """Export memory contents as JSON string."""
        return json.dumps(self.store, indent=2)


if __name__ == "__main__":
    mem = Memory()
    mem.add("status", "active", "sensor-A")
    mem.add("status", "active", "sensor-B")
    mem.add("status", "inactive", "sensor-C")
    print(mem.export())