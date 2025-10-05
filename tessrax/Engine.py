# tessrax_engine_v2_3.py
# Tessrax Engine v2.3 – Persistent Governance System
# Author: Joshua Vetos

import uuid, json, os, hashlib, datetime
from typing import List, Tuple, Dict, Any, Optional

# --- Utility ---
def now() -> str:
    """Return current UTC timestamp in ISO8601 with Z suffix."""
    return datetime.datetime.utcnow().isoformat() + "Z"

def sha256(data: Any) -> str:
    """Generate SHA256 hash of JSON-serializable data."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

# --- Tessrax Engine Class ---
class TessraxEngine:
    def __init__(self, claims_file="claims.jsonl", handoff_file="handoffs.jsonl"):
        self.engine_version = "2.3"
        self.author = "Joshua Vetos"
        self.claims_file = claims_file
        self.handoff_file = handoff_file
        os.makedirs(os.path.dirname(self.claims_file) or ".", exist_ok=True)

    # --- Fuel Metabolizer ---
    def _generate_fuel(self, pair: Tuple[str, str]) -> int:
        joined = "::".join(pair).encode("utf-8")
        digest = hashlib.sha256(joined).hexdigest()
        return int(digest, 16) % 100000

    def _metabolize(self, contradictions: List[Tuple[str, str]], source: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": str(uuid.uuid4()),
                "contradiction": {"side_a": pair[0], "side_b": pair[1]},
                "fuel": self._generate_fuel(pair),
                "timestamp": now(),
                "source": source,
                "status": "unresolved"
            }
            for pair in contradictions
        ]

    # --- Governance Claims ---
    def _sign_claim(self, agent: str, claim: str) -> Dict[str, Any]:
        ts = now()
        signature = hashlib.sha256(f"{agent}:{claim}:{ts}".encode()).hexdigest()
        return {"agent": agent, "claim": claim, "timestamp": ts, "signature": signature}

    def _verify_claim(self, entry: Dict[str, Any]) -> bool:
        recomputed = hashlib.sha256(
            f"{entry['agent']}:{entry['claim']}:{entry['timestamp']}".encode()
        ).hexdigest()
        return recomputed == entry["signature"]

    def _persist_claim(self, entry: Dict[str, Any]):
        with open(self.claims_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # --- Scar Registry ---
    def _log_scar(
        self,
        name: str,
        contradiction: Tuple[str, str],
        parent_id: Optional[str] = None,
        severity: str = "medium",
        status: str = "open",
        impact_score: int = 50
    ) -> Dict[str, Any]:
        return {
            "scar_id": str(uuid.uuid4()),
            "name": name,
            "contradiction": {"side_a": contradiction[0], "side_b": contradiction[1]},
            "parent_id": parent_id,
            "severity": severity,
            "impact_score": impact_score,
            "status": status,
            "logged_at": now()
        }

    def compute_scar_index(self, scars: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [s["impact_score"] for s in scars]
        return {
            "average": sum(scores) / len(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "min": min(scores) if scores else 0,
            "count": len(scores)
        }

    # --- Continuity Handoff ---
    def _handoff(
        self,
        state: Dict[str, Any],
        parent_id: Optional[str] = None,
        parent_hash: Optional[str] = None,
        officiant: str = "system",
        ritual_step: str = "transfer"
    ) -> Dict[str, Any]:
        chained_state = {"parent_hash": parent_hash, "state": state}
        state_hash = sha256(chained_state)
        return {
            "handoff_id": str(uuid.uuid4()),
            "parent_id": parent_id,
            "parent_hash": parent_hash,
            "state": state,
            "state_hash": state_hash,
            "officiant": officiant,
            "ritual_step": ritual_step,
            "timestamp": now()
        }

    def _persist_handoff(self, entry: Dict[str, Any]):
        with open(self.handoff_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _get_last_handoff(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.handoff_file): return None
        with open(self.handoff_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return json.loads(lines[-1]) if lines else None

    # --- Chain Verification ---
    def verify_handoff_chain(self) -> Dict[str, Any]:
        prev_hash = None
        report = []
        ok = True

        if not os.path.exists(self.handoff_file):
            return {"ok": False, "report": []}

        with open(self.handoff_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                entry = json.loads(line)
                chained = {"parent_hash": entry["parent_hash"], "state": entry.get("state")}
                recomputed = sha256(chained)
                valid = (recomputed == entry["state_hash"])
                linked = (entry["parent_hash"] == prev_hash or prev_hash is None)

                report.append({
                    "handoff": i,
                    "id": entry["handoff_id"],
                    "valid_hash": valid,
                    "linked": linked
                })
                if not (valid and linked): ok = False
                prev_hash = entry["state_hash"]

        return {"ok": ok, "report": report}

    # --- Replay ---
    def replay_last_run(self) -> Optional[Dict[str, Any]]:
        last = self._get_last_handoff()
        if not last: return None
        state = last.get("state", {})
        recomputed_hash = sha256({"parent_hash": last["parent_hash"], "state": state})
        drift = recomputed_hash != last["state_hash"]
        return {
            "last_handoff_id": last["handoff_id"],
            "drift_detected": drift,
            "recomputed_hash": recomputed_hash,
            "stored_hash": last["state_hash"],
            "state_summary": {k: len(v) if isinstance(v, list) else "n/a" for k, v in state.items()}
        }

    # --- Orchestrator ---
    def run(
        self,
        contradictions: List[Tuple[str, str]],
        claim_text: str,
        agent: str = "GPT",
        source: str = "demo"
    ) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())

        # Step 1: Fuel
        fuel_units = self._metabolize(contradictions, source)
        fuel_hash = sha256(fuel_units)

        # Step 2: Claim
        claim = self._sign_claim(agent, claim_text)
        self._persist_claim(claim)
        verified = self._verify_claim(claim)

        # Step 3: Scars
        scars = [
            self._log_scar(f"scar_{i}", (c[0], c[1]), parent_id=fuel_units[i]["id"], impact_score=(i+1)*10)
            for i, c in enumerate(contradictions)
        ]
        scar_index = self.compute_scar_index(scars)

        # Step 4: Handoff
        last_handoff = self._get_last_handoff()
        parent_id = last_handoff["handoff_id"] if last_handoff else None
        parent_hash = last_handoff["state_hash"] if last_handoff else None

        state = {"fuel_units": fuel_units, "scars": scars, "claims": [claim]}
        handoff = self._handoff(state, parent_id, parent_hash, officiant=agent)
        self._persist_handoff(handoff)

        # Step 5: Ledger
        ledger = [
            {"stage": "fuel", "count": len(fuel_units), "ids": [f["id"] for f in fuel_units], "hash": fuel_hash, "timestamp": now()},
            {"stage": "claim", "verified": verified, "id": claim["signature"], "timestamp": now()},
            {"stage": "scars", "count": len(scars), "scar_index": scar_index, "timestamp": now()},
            {"stage": "handoff", "id": handoff["handoff_id"], "hash": handoff["state_hash"], "timestamp": handoff["timestamp"]}
        ]

        return {
            "engine_version": self.engine_version,
            "author": self.author,
            "run_id": run_id,
            "run_timestamp": now(),
            "fuel_units": fuel_units,
            "governance_claim": claim,
            "verified": verified,
            "scars": scars,
            "scar_index": scar_index,
            "handoff": handoff,
            "ledger": ledger
        }

# --- Demo ---
if __name__ == "__main__":
    engine = TessraxEngine()
    sample_contradictions = [
        ("AI is safe", "AI is dangerous"),
        ("Humans are rational", "Humans are irrational")
    ]
    output = engine.run(sample_contradictions, "Transparency is a safeguard")
    print(json.dumps(output, indent=2))
    print("\nChain Verification:", json.dumps(engine.verify_handoff_chain(), indent=2))
    print("\nReplay:", json.dumps(engine.replay_last_run(), indent=2))
