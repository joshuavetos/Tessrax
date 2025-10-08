"""
Tessrax Governance Kernel v2.0
Central coordination service for enforcement, quorum voting, and event routing.
Now includes native handling for DESIGN_DECISION_RECORDED and POLICY_VIOLATION events.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

# ============================================================
# Setup
# ============================================================

LEDGER_PATH = Path("data/ledger.jsonl")
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility
# ============================================================

def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# ============================================================
# GovernanceKernel Class
# ============================================================

class GovernanceKernel:
    """
    Core governance orchestrator.
    - Hash-chained ledger
    - Event publishing & subscription
    - Quorum voting
    - Policy enforcement hooks
    """

    def __init__(self):
        self.quorum_rules = {
            "merge_vote": {"required": 2, "total": 3},
            "policy_violation_vote": {"required": 2, "total": 3},
        }
        self.subscribers = []
        self.ledger = []

        # Load existing ledger if present
        if LEDGER_PATH.exists():
            with open(LEDGER_PATH) as f:
                for line in f:
                    try:
                        self.ledger.append(json.loads(line))
                    except:
                        pass

    # ------------------------ Ledger -------------------------
    def append_event(self, event: dict):
        """Append event with hash chaining and persistence."""
        prev = self.ledger[-1]["hash"] if self.ledger else None
        event["prev_hash"] = prev
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()
        event["hash"] = "sha256:" + _sha256(event)
        self.ledger.append(event)

        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")

        self._notify_subscribers(event)
        self._auto_reactions(event)
        return event["hash"]

    # --------------------- Notifications ---------------------
    def register_subscriber(self, callback):
        """Register callback for all new events."""
        self.subscribers.append(callback)

    def _notify_subscribers(self, event):
        for cb in self.subscribers:
            try:
                cb(event)
            except Exception as e:
                print("Subscriber error:", e)

    # ---------------------- Quorum Logic ---------------------
    def quorum_vote(self, proposal_id: str, votes: list, vote_type: str = "merge_vote"):
        """
        votes = [{"peer":name, "decision":bool}, ...]
        Returns True if quorum achieved under rule.
        """
        rule = self.quorum_rules.get(vote_type, {"required": 1, "total": len(votes)})
        required = rule["required"]
        total = rule["total"]

        approvals = sum(1 for v in votes if v["decision"])
        passed = approvals >= required

        result = {
            "event": "QUORUM_VOTE_RESULT",
            "proposal_id": proposal_id,
            "vote_type": vote_type,
            "votes": votes,
            "passed": passed,
            "approved": approvals,
            "required": required,
            "total": total,
        }
        self.append_event(result)
        return passed

    # ============================================================
    # Built-in Auto-Reactions
    # ============================================================

    def _auto_reactions(self, event):
        """Handle known event types with built-in governance behavior."""

        # DESIGN_DECISION_RECORDED → automatically acknowledge & log
        if event["event"] == "DESIGN_DECISION_RECORDED":
            ack = {
                "event": "DESIGN_DECISION_ACK",
                "ack_of": event["hash"],
                "file_changed": event.get("file_changed"),
                "tags": event.get("tags", []),
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._append_internal(ack)
            print(f"🧱 Logged design decision for {event.get('file_changed')}")

        # POLICY_VIOLATION → trigger automatic quorum review
        elif event["event"] == "POLICY_VIOLATION":
            print("🚨 Policy violation detected, starting quorum review…")
            proposal_id = f"PV-{event['hash'][:8]}"
            votes = [
                {"peer": "governor_1", "decision": True},
                {"peer": "governor_2", "decision": True},
                {"peer": "governor_3", "decision": False},
            ]
            result = self.quorum_vote(proposal_id, votes, vote_type="policy_violation_vote")
            reaction = {
                "event": "POLICY_VIOLATION_REVIEWED",
                "proposal_id": proposal_id,
                "passed": result,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._append_internal(reaction)

    def _append_internal(self, event):
        """Helper to append internal system events without triggering recursion."""
        prev = self.ledger[-1]["hash"] if self.ledger else None
        event["prev_hash"] = prev
        event["hash"] = "sha256:" + _sha256(event)
        self.ledger.append(event)
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")

# ============================================================
# Demo Harness
# ============================================================

if __name__ == "__main__":
    kernel = GovernanceKernel()

    def printer(e): print("EVENT:", e["event"])
    kernel.register_subscriber(printer)

    # Demo 1 — normal proposal and vote
    kernel.append_event({
        "event": "PROPOSAL_SUBMITTED",
        "proposal_id": "P001",
        "details": "Merge branches A+B"
    })

    votes = [
        {"peer": "node1", "decision": True},
        {"peer": "node2", "decision": True},
        {"peer": "node3", "decision": False},
    ]
    kernel.quorum_vote("P001", votes)

    # Demo 2 — design decision from scaffolding engine
    kernel.append_event({
        "event": "DESIGN_DECISION_RECORDED",
        "file_changed": "scaffolding_engine.py",
        "tags": ["meta", "governance"],
    })

    # Demo 3 — trigger policy violation
    kernel.append_event({
        "event": "POLICY_VIOLATION",
        "file_changed": "fork_reconciliation_engine.py",
        "tags": ["fork", "governance"],
        "policy": "Simulated test of policy enforcement"
    })