"""
attention_governance_kernel.py
Tessrax v0.1 — Governance interface for attention-economy contradictions.
"""

from ledger import add_event, merkle_root
from quorum import check_quorum
from dissent import record_dissent

def propose_focus_policy(name: str, proposer: str, rationale: str):
    event = {"action": "propose", "policy": name, "proposer": proposer, "rationale": rationale}
    add_event(event)
    print(f"Policy proposed: {name}")

def vote_on_policy(votes: dict):
    result = check_quorum(votes)
    add_event({"action": "vote", "result": result, "votes": votes})
    if not result:
        record_dissent("attention", "Focus policy quorum not met")
    print("Vote:", "Accepted" if result else "Rejected")

def summarize():
    print("Merkle Root:", merkle_root())

if __name__ == "__main__":
    propose_focus_policy("Calm-Time Reward System", "user-council",
                         "Reward platforms that reduce compulsive engagement loops.")
    votes = {"a1": "yes", "a2": "yes", "a3": "no", "a4": "yes", "a5": "yes", "a6": "yes", "a7": "no"}
    vote_on_policy(votes)
    summarize()