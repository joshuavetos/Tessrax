"""
Tessrax Governance Kernel
Central coordination service for enforcement, quorum voting, and event routing.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json
from datetime import datetime
from pathlib import Path
import hashlib

LEDGER_PATH = Path("data/ledger.jsonl")

def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# ============================================================
# GovernanceKernel
# ============================================================

class GovernanceKernel:
    def __init__(self):
        self.quorum_rules = {"merge_vote": {"required": 2, "total": 3}}
        self.subscribers = []
        self.ledger = []
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------ Ledger -------------------------
    def append_event(self, event:dict):
        """Append event with hash chaining and persist."""
        prev = self.ledger[-1]["hash"] if self.ledger else None
        event["prev_hash"] = prev
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()
        event["hash"] = "sha256:" + _sha256(event)
        self.ledger.append(event)
        with open(LEDGER_PATH,"a") as f:
            f.write(json.dumps(event) + "\n")
        self._notify_subscribers(event)
        return event["hash"]

    # --------------------- Governance ------------------------
    def register_subscriber(self, callback):
        """Register callback to receive new events."""
        self.subscribers.append(callback)

    def _notify_subscribers(self, event):
        for cb in self.subscribers:
            try:
                cb(event)
            except Exception as e:
                print("Subscriber error:", e)

    def quorum_vote(self, proposal_id:str, votes:list):
        """
        votes = list of {"peer":name,"decision":bool}
        Returns True if quorum achieved.
        """
        total = self.quorum_rules["merge_vote"]["total"]
        required = self.quorum_rules["merge_vote"]["required"]
        approvals = sum(1 for v in votes if v["decision"])
        passed = approvals >= required
        result = {
            "event": "QUORUM_VOTE_RESULT",
            "proposal_id": proposal_id,
            "votes": votes,
            "passed": passed,
            "approved": approvals,
            "total": total,
        }
        self.append_event(result)
        return passed

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    kernel = GovernanceKernel()

    # Register a simple print subscriber
    kernel.register_subscriber(lambda e: print("EVENT:", e["event"]))

    # Log a sample proposal
    kernel.append_event({"event":"PROPOSAL_SUBMITTED","proposal_id":"P001","details":"Merge branches A+B"})

    # Simulate quorum voting
    votes = [
        {"peer":"node1","decision":True},
        {"peer":"node2","decision":True},
        {"peer":"node3","decision":False}
    ]
    passed = kernel.quorum_vote("P001", votes)
    print("Vote passed?", passed)