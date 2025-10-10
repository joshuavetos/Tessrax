"""
housing_governance_kernel.py
Tessrax v0.1 — Governance interface for housing contradiction management.
"""

import json
from dissent import record_dissent
from quorum import check_quorum
from ledger import add_event, merkle_root

def propose_primitive(name: str, proposer: str, rationale: str):
    event = {"action": "propose", "primitive": name, "proposer": proposer, "rationale": rationale}
    add_event(event)
    print(f"Proposal logged: {name}")

def vote_on_primitive(votes: dict):
    result = check_quorum(votes)
    add_event({"action": "vote", "result": result, "votes": votes})
    if not result:
        record_dissent("governance", "Quorum not met")
    print("Vote result:", "Accepted" if result else "Rejected")

def summarize_governance():
    print("Merkle Root:", merkle_root())

if __name__ == "__main__":
    propose_primitive("Durability Yield Calculator", "builder-alpha", "Reward long-lived structures.")
    votes = {"a1": "yes", "a2": "yes", "a3": "no", "a4": "yes", "a5": "yes", "a6": "no", "a7": "yes"}
    vote_on_primitive(votes)
    summarize_governance()