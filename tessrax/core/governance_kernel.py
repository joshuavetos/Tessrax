"""
Tessrax Governance Kernel v3.0
Now dynamically enforces POLICY_RULES from policy_rules.py.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from policy_rules import POLICY_RULES

LEDGER_PATH = Path("data/ledger.jsonl")
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

class GovernanceKernel:
    """Central orchestrator for Tessrax governance events and enforcement."""

    def __init__(self):
        self.subscribers = []
        self.ledger = []
        if LEDGER_PATH.exists():
            with open(LEDGER_PATH) as f:
                for line in f:
                    try:
                        self.ledger.append(json.loads(line))
                    except:
                        pass
        self.quorum_rules = {"default": {"required": 2, "total": 3}}

    # --------------- Core Append -----------------
    def append_event(self, event):
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

    # --------------- Notifications -----------------
    def register_subscriber(self, callback):
        self.subscribers.append(callback)

    def _notify_subscribers(self, event):
        for cb in self.subscribers:
            try:
                cb(event)
            except Exception as e:
                print("Subscriber error:", e)

    # --------------- Policy + Reaction Engine -----------------
    def _auto_reactions(self, event):
        """Handle known event types and enforce policies."""
        if event["event"] == "DESIGN_DECISION_RECORDED":
            self._check_policies(event)
            ack = {
                "event": "DESIGN_DECISION_ACK",
                "ack_of": event["hash"],
                "file_changed": event.get("file_changed"),
                "tags": event.get("tags", []),
                "timestamp": datetime.utcnow().isoformat()
            }
            self._append_internal(ack)
            print(f"🧱 Acknowledged design decision for {event.get('file_changed')}")

        elif event["event"] == "POLICY_VIOLATION":
            print("🚨 Policy violation detected — triggering quorum vote")
            proposal_id = f"PV-{event['hash'][:8]}"
            votes = [
                {"peer": "gov1", "decision": True},
                {"peer": "gov2", "decision": True},
                {"peer": "gov3", "decision": False},
            ]
            result = self._quorum_vote(proposal_id, votes)
            reaction = {
                "event": "POLICY_VIOLATION_REVIEWED",
                "proposal_id": proposal_id,
                "passed": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            self._append_internal(reaction)

    def _check_policies(self, event):
        """Evaluate POLICY_RULES for each DESIGN_DECISION_RECORDED event."""
        file_changed = event.get("file_changed", "")
        tags = set(event.get("tags", []))

        for rule_name, rule in POLICY_RULES.items():
            if re.match(rule["pattern"], file_changed):
                missing = [t for t in rule["required_tags"] if t not in tags]
                if missing:
                    violation = {
                        "event": "POLICY_VIOLATION",
                        "policy": rule_name,
                        "missing_tags": missing,
                        "file_changed": file_changed,
                        "enforcement": rule["enforcement"],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    self._append_internal(violation)
                    print(f"⚠️ {rule_name}: Missing tags {missing} in {file_changed}")

                    # optional escalation
                    if rule["enforcement"] == "reject":
                        raise Exception(f"Rejected by policy: {rule_name}")
                    elif rule["enforcement"] == "quorum":
                        self.append_event(violation)

    def _append_internal(self, event):
        """Internal append without recursive reactions."""
        prev = self.ledger[-1]["hash"] if self.ledger else None
        event["prev_hash"] = prev
        event["hash"] = "sha256:" + _sha256(event)
        self.ledger.append(event)
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")

    # --------------- Quorum Voting -----------------
    def _quorum_vote(self, proposal_id, votes):
        required = self.quorum_rules["default"]["required"]
        approvals = sum(1 for v in votes if v["decision"])
        passed = approvals >= required
        vote_event = {
            "event": "QUORUM_VOTE_RESULT",
            "proposal_id": proposal_id,
            "approved": approvals,
            "required": required,
            "passed": passed,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._append_internal(vote_event)
        return passed

# --------------- Demo Harness -----------------
if __name__ == "__main__":
    kernel = GovernanceKernel()
    kernel.append_event({
        "event": "DESIGN_DECISION_RECORDED",
        "file_changed": "fork_reconciliation_engine.py",
        "tags": ["governance"],
    })