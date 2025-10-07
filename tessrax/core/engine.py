# tessrax_engine.py
# Tessrax Engine v4.0 — Modern Rewrite for Governance & Contradiction Metabolism
# Author: Joshua Vetos (rewritten by OpenAI GPT-4o)
# License: Creative Commons Attribution 4.0 International

import os
import json
import uuid
import hashlib
import datetime
import threading
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Union, Literal

try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False


# ================================
# Utility Functions
# ================================

def now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256(data: Union[str, Dict]) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def atomic_append(path: str, entry: Dict, lock: threading.Lock):
    ensure_dir(path)
    if HAS_FILELOCK:
        with FileLock(path + ".lock"):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
    else:
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")


# ================================
# Entry Data Classes
# ================================

@dataclass
class Claim:
    id: str
    agent: str
    claim: str
    timestamp: str
    signature: str

@dataclass
class Scar:
    scar_id: str
    contradiction: Dict[str, str]
    fuel: int
    severity: str
    status: Literal["open", "closed"]
    source: str
    timestamp: str

@dataclass
class Handoff:
    handoff_id: str
    from_agent: str
    to_agent: str
    topic: str
    timestamp: str

@dataclass
class IntegrityLink:
    timestamp: str
    entry_type: str
    prev_hash: Optional[str]
    current_hash: str


# ================================
# File-backed Append Store
# ================================

class LogStore:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        ensure_dir(path)

    def append(self, entry: Dict):
        atomic_append(self.path, entry, self.lock)

    def load_all(self) -> List[Dict]:
        return load_jsonl(self.path)

    def overwrite_all(self, entries: List[Dict]):
        ensure_dir(self.path)
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")


# ================================
# Tessrax Engine
# ================================

class TessraxEngine:
    def __init__(self, base_dir: str = "data", verbose: bool = False):
        self.version = "4.0"
        self.verbose = verbose
        self.stores = {
            "claims": LogStore(os.path.join(base_dir, "claims.jsonl")),
            "scars": LogStore(os.path.join(base_dir, "scars.jsonl")),
            "handoffs": LogStore(os.path.join(base_dir, "handoffs.jsonl")),
            "integrity": LogStore(os.path.join(base_dir, "integrity_chain.jsonl")),
        }

    def log(self, message: str):
        if self.verbose:
            print(f"[Tessrax v{self.version}] {now()} — {message}")

    # -----------------------
    # Claims
    # -----------------------
    def sign_claim(self, agent: str, claim_text: str) -> Dict:
        timestamp = now()
        signature = sha256(f"{agent}:{claim_text}:{timestamp}")
        entry = Claim(
            id=str(uuid.uuid4()),
            agent=agent,
            claim=claim_text,
            timestamp=timestamp,
            signature=signature
        )
        self._commit("claims", entry)
        self.log(f"Signed claim from '{agent}': {claim_text[:50]}")
        return asdict(entry)

    def verify_claim(self, entry: Dict) -> bool:
        expected_sig = sha256(f"{entry['agent']}:{entry['claim']}:{entry['timestamp']}")
        return expected_sig == entry.get("signature")

    def get_claims_by_agent(self, agent: str) -> List[Dict]:
        return [c for c in self.stores["claims"].load_all() if c.get("agent") == agent]

    # -----------------------
    # Scars
    # -----------------------
    def metabolize(self, contradictions: List[tuple], source: str) -> List[Dict]:
        scars = []
        for side_a, side_b in contradictions:
            fuel = int(sha256(f"{side_a}::{side_b}"), 16) % 100_000
            scar = Scar(
                scar_id=str(uuid.uuid4()),
                contradiction={"side_a": side_a, "side_b": side_b},
                fuel=fuel,
                severity=self._severity_level(fuel),
                status="open",
                source=source,
                timestamp=now()
            )
            self._commit("scars", scar)
            scars.append(asdict(scar))
        self.log(f"Metabolized {len(scars)} contradictions from '{source}'")
        return scars

    def scars_summary(self) -> Dict:
        scars = self.stores["scars"].load_all()
        if not scars:
            return {"count": 0, "avg_fuel": 0, "volatility": 0, "severities": []}
        fuels = [s["fuel"] for s in scars]
        return {
            "count": len(scars),
            "avg_fuel": round(sum(fuels) / len(fuels), 2),
            "volatility": max(fuels) - min(fuels),
            "severities": sorted({s["severity"] for s in scars}),
        }

    def _severity_level(self, fuel: int) -> str:
        if fuel > 80000:
            return "critical"
        elif fuel > 50000:
            return "high"
        elif fuel > 20000:
            return "medium"
        return "low"

    # -----------------------
    # Handoffs
    # -----------------------
    def handoff(self, from_agent: str, to_agent: str, topic: str) -> Dict:
        entry = Handoff(
            handoff_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            timestamp=now()
        )
        self._commit("handoffs", entry)
        self.log(f"Handoff recorded: {from_agent} → {to_agent} on '{topic}'")
        return asdict(entry)

    # -----------------------
    # Integrity Chain
    # -----------------------
    def _commit(self, kind: str, entry_obj: Any):
        entry_dict = asdict(entry_obj)
        self.stores[kind].append(entry_dict)
        self._update_integrity(entry_obj)

    def _update_integrity(self, entry_obj: Any):
        chain = self.stores["integrity"].load_all()
        prev_hash = chain[-1]["current_hash"] if chain else None
        current_hash = sha256(asdict(entry_obj))
        link = IntegrityLink(
            timestamp=now(),
            entry_type=self._entry_type(entry_obj),
            prev_hash=prev_hash,
            current_hash=current_hash
        )
        self.stores["integrity"].append(asdict(link))

    def _entry_type(self, entry: Any) -> str:
        if isinstance(entry, Claim):
            return "claim"
        elif isinstance(entry, Scar):
            return "scar"
        elif isinstance(entry, Handoff):
            return "handoff"
        return "unknown"

    def verify_integrity_chain(self) -> bool:
        chain = self.stores["integrity"].load_all()
        for i in range(1, len(chain)):
            if chain[i]["prev_hash"] != chain[i - 1]["current_hash"]:
                self.log("⚠️ Integrity chain broken at index {i}")
                return False
        return True

    # -----------------------
    # State Snapshot
    # -----------------------
    def export_state(self) -> Dict[str, List[Dict]]:
        return {k: store.load_all() for k, store in self.stores.items() if k != "integrity"}

    def import_state(self, state: Dict[str, List[Dict]]):
        for kind, entries in state.items():
            if kind in self.stores:
                self.stores[kind].overwrite_all(entries)
        self.log("State successfully imported.")

# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    engine = TessraxEngine(verbose=True)
    engine.sign_claim("Alice", "The Earth orbits the Sun.")
    engine.handoff("Alice", "Bob", "Solar responsibility")
    engine.metabolize([("The Earth orbits the Sun", "The Sun orbits the Earth")], source="demo")
    print(engine.scars_summary())
    print("Integrity chain valid:", engine.verify_integrity_chain())