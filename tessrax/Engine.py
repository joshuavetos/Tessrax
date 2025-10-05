# tessrax_engine_v3_0.py
# Tessrax Engine v3.0 — Persistent Governance and Contradiction Metabolism System
# Author: Joshua Vetos
# License: Creative Commons Attribution 4.0 International

import os
import json
import uuid
import hashlib
import datetime
import threading
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False


# === Utility Functions ========================================================

def now() -> str:
    """Return current UTC time in ISO8601 format (Z-suffixed)."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serializable data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def ensure_dir(path: str):
    """Ensure directory exists for the given path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def load_jsonl(path: str) -> List[dict]:
    """Load all entries from a JSONL file."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def atomic_write(path: str, entry: dict, lock_map: Dict[str, threading.Lock]):
    """Thread- and process-safe append to JSONL file."""
    ensure_dir(path)
    lock = lock_map.setdefault(path, threading.Lock())

    if HAS_FILELOCK:
        with FileLock(path + ".lock"):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    else:
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")


# === Core Engine ===============================================================

class TessraxEngine:
    """
    Tessrax Engine v3.0
    A self-persisting, contradiction-metabolizing governance kernel.

    - Handles claims (assertions)
    - Handles scars (contradictions)
    - Supports auditability and resilience
    """

    def __init__(
        self,
        base_dir: str = ".",
        verbose: bool = False
    ):
        self.version = "3.0"
        self.author = "Joshua Vetos"
        self.verbose = verbose
        self.paths = {
            "claims": os.path.join(base_dir, "claims.jsonl"),
            "handoffs": os.path.join(base_dir, "handoffs.jsonl"),
            "scars": os.path.join(base_dir, "scars.jsonl"),
            "integrity": os.path.join(base_dir, "integrity_chain.jsonl")
        }
        for p in self.paths.values():
            ensure_dir(p)
        self._locks = {p: threading.Lock() for p in self.paths.values()}

    # --------------------------------------------------------------------------
    # Logging and Status
    # --------------------------------------------------------------------------
    def log(self, msg: str):
        if self.verbose:
            print(f"[Tessrax v{self.version}] {now()} — {msg}")

    # --------------------------------------------------------------------------
    # Claim System
    # --------------------------------------------------------------------------
    def sign_claim(self, agent: str, claim: str) -> Dict[str, Any]:
        """Sign and timestamp a claim."""
        timestamp = now()
        signature = hashlib.sha256(f"{agent}:{claim}:{timestamp}".encode()).hexdigest()
        entry = {
            "id": str(uuid.uuid4()),
            "agent": agent,
            "claim": claim,
            "timestamp": timestamp,
            "signature": signature
        }
        self._persist("claims", entry)
        self._update_integrity(entry)
        self.log(f"Signed claim: {agent} → '{claim[:40]}'")
        return entry

    def verify_claim(self, entry: Dict[str, Any]) -> bool:
        """Verify claim integrity."""
        expected = hashlib.sha256(
            f"{entry['agent']}:{entry['claim']}:{entry['timestamp']}".encode()
        ).hexdigest()
        valid = expected == entry["signature"]
        if not valid:
            self.log(f"❌ Claim verification failed for agent '{entry['agent']}'")
        return valid

    def claims_by_agent(self, agent: str) -> List[Dict[str, Any]]:
        """Retrieve all claims by a specific agent."""
        return [c for c in load_jsonl(self.paths["claims"]) if c.get("agent") == agent]

    def latest_claim(self) -> Optional[Dict[str, Any]]:
        """Return the most recent claim by timestamp."""
        claims = load_jsonl(self.paths["claims"])
        return max(claims, key=lambda c: c["timestamp"], default=None)

    # --------------------------------------------------------------------------
    # Scar (Contradiction) System
    # --------------------------------------------------------------------------
    def metabolize(self, contradictions: List[Tuple[str, str]], source: str) -> List[Dict[str, Any]]:
        """Transform contradictions into persistent scars."""
        scars = []
        for a, b in contradictions:
            fuel = int(hashlib.sha256(f"{a}::{b}".encode()).hexdigest(), 16) % 100000
            entry = {
                "scar_id": str(uuid.uuid4()),
                "contradiction": {"side_a": a, "side_b": b},
                "fuel": fuel,
                "status": "open",
                "severity": self._determine_severity(fuel),
                "source": source,
                "timestamp": now()
            }
            scars.append(entry)
            self._persist("scars", entry)
        self.log(f"Metabolized {len(scars)} contradictions from '{source}'.")
        return scars

    def _determine_severity(self, fuel: int) -> str:
        if fuel > 80000:
            return "critical"
        elif fuel > 50000:
            return "high"
        elif fuel > 20000:
            return "medium"
        return "low"

    def scars_summary(self) -> Dict[str, Any]:
        """Compute volatility metrics and averages for scars."""
        scars = load_jsonl(self.paths["scars"])
        count = len(scars)
        if not scars:
            return {"count": 0, "avg_fuel": 0, "volatility": 0}
        fuels = [s["fuel"] for s in scars]
        volatility = max(fuels) - min(fuels)
        avg_fuel = sum(fuels) / count
        severities = {s["severity"] for s in scars}
        return {
            "count": count,
            "avg_fuel": round(avg_fuel, 2),
            "volatility": volatility,
            "severities": list(severities)
        }

    # --------------------------------------------------------------------------
    # Handoffs (Delegations / Transfers)
    # --------------------------------------------------------------------------
    def handoff(self, from_agent: str, to_agent: str, topic: str) -> Dict[str, Any]:
        """Record a responsibility handoff event."""
        entry = {
            "handoff_id": str(uuid.uuid4()),
            "from": from_agent,
            "to": to_agent,
            "topic": topic,
            "timestamp": now()
        }
        self._persist("handoffs", entry)
        self._update_integrity(entry)
        self.log(f"Handoff: {from_agent} → {to_agent} ({topic})")
        return entry

    # --------------------------------------------------------------------------
    # State & Integrity
    # --------------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        """Export complete state snapshot."""
        return {k: load_jsonl(v) for k, v in self.paths.items() if k != "integrity"}

    def import_state(self, state: Dict[str, Any]):
        """Replace all persistent files with provided state."""
        for name, entries in state.items():
            if name not in self.paths:
                continue
            with open(self.paths[name], "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
        self.log("State imported successfully.")

    def _update_integrity(self, entry: dict):
        """Update rolling integrity chain for tamper detection."""
        chain = load_jsonl(self.paths["integrity"])
        prev_hash = chain[-1]["current_hash"] if chain else None
        current_hash = sha256(entry)
        link = {
            "timestamp": now(),
            "entry_type": self._infer_type(entry),
            "prev_hash": prev_hash,
            "current_hash": current_hash
        }
        atomic_write(self.paths["integrity"], link, self._locks)

    def verify_integrity_chain(self) -> bool:
        """Check hash-chain continuity."""
        chain = load_jsonl(self.paths["integrity"])
        for i in range(1, len(chain)):
            if chain[i]["prev_hash"] != chain[i - 1]["current_hash"]:
                self.log("Integrity chain broken.")
                return False
        return True

    def _infer_type(self, entry: dict) -> str:
        if "claim" in entry:
            return "claim"
        if "scar_id" in entry:
            return "scar"
        if "handoff_id" in entry:
            return "handoff"
        return "unknown"

    # --------------------------------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------------------------------
    def _persist(self, kind: str, entry: dict):
        path = self.paths[kind]
        atomic_write(path, entry, self._locks)

# ------------------------------------------------------------------------------
# Example usage (if run directly)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    engine = TessraxEngine(verbose=True)
    engine.sign_claim("AgentX", "Truth emerges from contradiction.")
    engine.metabolize([("A is true", "A is false")], source="test")
    engine.handoff("AgentX", "AgentY", "Mediation of paradox")
    print(engine.scars_summary())
    print("Integrity verified:", engine.verify_integrity_chain())