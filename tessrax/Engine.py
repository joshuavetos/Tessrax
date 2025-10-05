```python name=tessrax/Engine.py url=https://github.com/joshuavetos/Tessrax/blob/38bef2b0a12af67e35504eee7c6433e8a294db12/tessrax/Engine.py
# tessrax_engine_v2_4.py
# Tessrax Engine v2.4 – Persistent Governance System (Enhanced)
# Author: Joshua Vetos

import uuid
import json
import os
import hashlib
import datetime
import threading
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

# --- Utility Functions ---

def now() -> str:
    """Return current UTC timestamp in ISO8601 format with Z suffix."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256(data: Any) -> str:
    """Generate a SHA256 hash of JSON-serializable data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def atomic_append(filename: str, entry: dict):
    """Atomically append a JSON entry to a JSONL file, with basic file locking."""
    lock = threading.Lock()
    with lock:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def safe_load_jsonl(filename: str) -> List[dict]:
    """Safely load all JSON objects from a JSONL file."""
    if not os.path.exists(filename):
        return []
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def ensure_dir_exists(path: str):
    """Ensure the directory for the given file path exists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# --- Tessrax Engine Class ---

class TessraxEngine:
    """
    Core governance engine for Tessrax.
    Handles claims, contradiction metabolism, and scar registry with simple persistence.
    Enhanced for thread-safety, robustness, and usability.
    """

    def __init__(
        self,
        claims_file: str = "claims.jsonl",
        handoff_file: str = "handoffs.jsonl",
        scars_file: str = "scars.jsonl",
        verbose: bool = False
    ):
        self.engine_version = "2.4"
        self.author = "Joshua Vetos"
        self.claims_file = claims_file
        self.handoff_file = handoff_file
        self.scars_file = scars_file
        self.verbose = verbose
        for path in [claims_file, handoff_file, scars_file]:
            ensure_dir_exists(path)
        self._file_locks: Dict[str, threading.Lock] = {
            path: threading.Lock() for path in [claims_file, handoff_file, scars_file]
        }

    def log(self, msg: str):
        if self.verbose:
            print(f"[TessraxEngine] {now()} - {msg}")

    # --- Contradiction Metabolism ---

    def _generate_fuel(self, pair: Tuple[str, str]) -> int:
        """Generate a pseudorandom 'fuel' score from a contradiction pair."""
        joined = "::".join(pair).encode("utf-8")
        digest = hashlib.sha256(joined).hexdigest()
        return int(digest, 16) % 100000

    def metabolize(
        self,
        contradictions: List[Tuple[str, str]],
        source: str,
        persist: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Transform contradictions into metabolized fuel records.
        Optionally persists the result to scars file.
        """
        records = []
        for pair in contradictions:
            entry = {
                "id": str(uuid.uuid4()),
                "contradiction": {"side_a": pair[0], "side_b": pair[1]},
                "fuel": self._generate_fuel(pair),
                "timestamp": now(),
                "source": source,
                "status": "unresolved"
            }
            records.append(entry)
            if persist:
                self._atomic_append_locked(self.scars_file, entry)
        self.log(f"Metabolized {len(records)} contradictions from source '{source}'.")
        return records

    # --- Governance Claims ---

    def sign_claim(self, agent: str, claim: str) -> Dict[str, Any]:
        """
        Sign a claim with agent, claim, timestamp, and SHA256 signature.
        """
        ts = now()
        signature = hashlib.sha256(f"{agent}:{claim}:{ts}".encode()).hexdigest()
        entry = {"agent": agent, "claim": claim, "timestamp": ts, "signature": signature}
        self.log(f"Signed claim for agent '{agent}': '{claim[:32]}'...")
        return entry

    def verify_claim(self, entry: Dict[str, Any]) -> bool:
        """
        Verify a claim's integrity by recomputing its signature.
        """
        try:
            recomputed = hashlib.sha256(
                f"{entry['agent']}:{entry['claim']}:{entry['timestamp']}".encode()
            ).hexdigest()
            valid = recomputed == entry["signature"]
            if not valid:
                self.log(f"Claim verification failed for agent '{entry.get('agent')}'.")
            return valid
        except Exception as e:
            self.log(f"Error verifying claim: {e}")
            return False

    def persist_claim(self, entry: Dict[str, Any]) -> None:
        """
        Persist a signed claim to the claims file.
        """
        self._atomic_append_locked(self.claims_file, entry)
        self.log(f"Persisted claim for agent '{entry.get('agent')}'.")

    def all_claims(self) -> List[Dict[str, Any]]:
        """
        Retrieve all claims from the claims file.
        """
        return safe_load_jsonl(self.claims_file)

    # --- Handoff Registry (placeholder for future use) ---

    def persist_handoff(self, entry: Dict[str, Any]) -> None:
        """
        Persist a handoff entry (future feature).
        """
        self._atomic_append_locked(self.handoff_file, entry)
        self.log("Persisted handoff entry.")

    def all_handoffs(self) -> List[Dict[str, Any]]:
        return safe_load_jsonl(self.handoff_file)

    # --- Scar Registry ---

    def log_scar(
        self,
        name: str,
        contradiction: Tuple[str, str],
        parent_id: Optional[str] = None,
        severity: str = "medium",
        status: str = "open",
        impact_score: int = 50,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Log a scar (record of contradiction) with metadata.
        Optionally persists to scars file.
        """
        entry = {
            "scar_id": str(uuid.uuid4()),
            "name": name,
            "contradiction": {"side_a": contradiction[0], "side_b": contradiction[1]},
            "parent_id": parent_id,
            "severity": severity,
            "impact_score": impact_score,
            "status": status,
            "logged_at": now()
        }
        if persist:
            self._atomic_append_locked(self.scars_file, entry)
            self.log(f"Logged scar '{name}' ({severity}), impact {impact_score}.")
        return entry

    def all_scars(self) -> List[Dict[str, Any]]:
        return safe_load_jsonl(self.scars_file)

    def compute_scar_index(self, scars: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Compute summary statistics for a list of scars.
        """
        scars = scars if scars is not None else self.all_scars()
        scores = [s["impact_score"] for s in scars if "impact_score" in s]
        count = len(scores)
        avg = sum(scores) / count if count else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        self.log(f"Computed scar index: count={count}, avg={avg:.1f}.")
        return {
            "count": count,
            "avg_impact": avg,
            "max_impact": max_score,
            "min_impact": min_score
        }

    # --- Internal Helpers ---

    def _atomic_append_locked(self, filename: str, entry: dict):
        """
        Thread-safe append to a JSONL file.
        """
        lock = self._file_locks.get(filename)
        if lock is None:
            lock = threading.Lock()
            self._file_locks[filename] = lock
        with lock:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    # --- Export/Import ---

    def export_state(self) -> Dict[str, Any]:
        """
        Export all persisted state as a dictionary.
        """
        return {
            "claims": self.all_claims(),
            "handoffs": self.all_handoffs(),
            "scars": self.all_scars()
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """
        Import state from a dictionary (overwrites files).
        """
        for name, entries, path in [
            ("claims", state.get("claims", []), self.claims_file),
            ("handoffs", state.get("handoffs", []), self.handoff_file),
            ("scars", state.get("scars", []), self.scars_file)
        ]:
            lock = self._file_locks[path]
            with lock:
                with open(path, "w", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")
            self.log(f"Imported {len(entries)} {name} entries.")

    # --- Utility / Query ---

    def find_scars_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Find all scars with a given status.
        """
        scars = self.all_scars()
        return [s for s in scars if s.get("status") == status]

    def find_claims_by_agent(self, agent: str) -> List[Dict[str, Any]]:
        """
        Find all claims by a given agent.
        """
        claims = self.all_claims()
        return [c for c in claims if c.get("agent") == agent]

    def get_latest_claim(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest claim by timestamp.
        """
        claims = self.all_claims()
        if not claims:
            return None
        return max(claims, key=lambda c: c["timestamp"])

    # --- End of TessraxEngine ---

```
