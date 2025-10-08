"""
Tessrax Governance Kernel v3.1
Autonomous event bus and policy enforcement engine.
"""

import json, re, hashlib
from datetime import datetime
from pathlib import Path
from policy_rules import POLICY_RULES

LEDGER_PATH = Path("data/ledger.jsonl")
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

def _sha256(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

class GovernanceKernel:
    def __init__(self):
        self.subscribers = []
        self.ledger = []
        if LEDGER_PATH.exists():
            with open(LEDGER_PATH) as f:
                for line in f:
                    try: self.ledger.append(json.loads(line))
                    except: pass
        self.quorum_rules = {"default": {"required": 2, "total": 3}}

    # ============================================================
    # Append + Notifications
    # ============================================================

    def append_event(self, event):
        prev = self.ledger[-1]["hash"] if self.ledger else None
        event["prev_hash"] = prev
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()
        event["hash"] = "sha256:" + _sha256(event)
        self.ledger.append(event)
        with open(LEDGER_PATH, "a") as f: f.write(json.dumps(event) + "\n")
        self._notify_subscribers(event)
        self._auto_reactions(event)
        return event["hash"]

    def register_subscriber(self, callback): self.subscribers.append(callback)
    def _notify_subscribers(self, event):
        for cb in self.subscribers:
            try: cb(event)
            except Exception as e: print("Subscriber error:", e)

    # ============================================================
    # Auto-Reactions
    # ============================================================

    def _auto_reactions(self, event):
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
            print("🚨 Policy violation detected → initiating quorum review")
            proposal_id = f"PV-{event['hash'][:8]}"
            votes = [
                {"peer": "gov1", "decision": True},
                {"peer": "gov2", "decision": True},
                {"peer": "gov3", "decision": False},
            ]
            result = self._quorum_vote(proposal_id, votes)
            review = {
                "event": "POLICY_VIOLATION_REVIEWED",
                "proposal_id": proposal_id,
                "passed": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            self._append_internal(review)

    # ============================================================
    # Policy Evaluation
    # ============================================================

    def _check_policies(self, event):
        file_changed = event.get("file_changed", "")
        tags = set(event.get("tags", []))

        for name, rule in POLICY_RULES.items():
            if re.match(rule["pattern"], file_changed):
                missing = [t for t in rule["required_tags"] if t not in tags]
                if missing:
                    violation = {
                        "event": "POLICY_VIOLATION",
                        "policy": name,
                        "missing_tags": missing,
                        "file_changed": file_changed,
                        "enforcement": rule["enforcement"],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    self._append_internal(violation)
                    print(f"⚠️ {name}: Missing {missing} in {file_changed}")

                    if rule["enforcement"] == "reject":
                        raise Exception(f"Rejected by policy: {name}")
                    elif rule["enforcement"] == "quorum":
                        self.append_event(violation)

    # ============================================================
    # Ledger Utilities
    # ============================================================

    def _append_internal(self, event):
        prev = self.ledger[-1]["hash"] if self.ledger else None
        event["prev_hash"] = prev
        event["hash"] = "sha256:" + _sha256(event)
        self.ledger.append(event)
        with open(LEDGER_PATH, "a") as f: f.write(json.dumps(event) + "\n")

    def _quorum_vote(self, proposal_id, votes):
        rule = self.quorum_rules["default"]
        approvals = sum(1 for v in votes if v["decision"])
        passed = approvals >= rule["required"]
        vote = {
            "event": "QUORUM_VOTE_RESULT",
            "proposal_id": proposal_id,
            "approved": approvals,
            "required": rule["required"],
            "total": rule["total"],
            "passed": passed,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._append_internal(vote)
        return passed

# ============================================================
# Demo Harness
# ============================================================

if __name__ == "__main__":
    kernel = GovernanceKernel()
    kernel.append_event({
        "event": "DESIGN_DECISION_RECORDED",
        "file_changed": "fork_reconciliation_engine.py",
        "tags": ["governance"]
    })