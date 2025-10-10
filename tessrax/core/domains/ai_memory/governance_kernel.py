"""
memory_governance_kernel.py
Tessrax v0.1 — Governance interface for AI memory contradiction management.
"""

from ledger import add_event, merkle_root
from quorum import check_quorum
from dissent import record_dissent

def propose_retention_policy(name: str, proposer: str, rationale: str):
    event = {"action": "propose", "policy": name, "proposer": proposer, "rationale": rationale}
    add_event(event)
    print(f"Policy proposed: {name}")

def vote_on_policy(votes: dict):
    result = check_quorum(votes)
    add_event({"action": "vote", "result": result, "votes": votes})
    if not result:
        record_dissent("ai_memory", "Retention policy quorum not met")
    print("Vote:", "Accepted" if result else "Rejected")

def summarize():
    print("Merkle Root:", merkle_root())

if __name__ == "__main__":
    propose_retention_policy("Provenance-First Retention", "researcher-01",
                             "Preserve contradictory entries for retraining transparency.")
    votes = {"a1": "yes", "a2": "no", "a3": "yes", "a4": "yes", "a5": "yes", "a6": "no", "a7": "yes"}
    vote_on_policy(votes)
    summarize()