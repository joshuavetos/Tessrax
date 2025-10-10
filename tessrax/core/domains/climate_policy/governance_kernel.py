"""
Governance kernel for climate contradictions.
"""
from ledger import add_event,merkle_root
from quorum import check_quorum
from dissent import record_dissent

def propose_policy(name,proposer,rationale):
    add_event({"action":"propose","policy":name,"proposer":proposer,"rationale":rationale})
    print(f"Proposed climate policy: {name}")

def vote(votes):
    res=check_quorum(votes)
    add_event({"action":"vote","result":res,"votes":votes})
    if not res:record_dissent("climate","Policy quorum not met")
    print("Vote:", "Accepted" if res else "Rejected")

def summary():
    print("Merkle Root:",merkle_root())

if __name__=="__main__":
    propose_policy("Green GDP Adjustment","policy-lab",
                   "Tie GDP bonuses to verified decarbonization yield.")
    votes={"a1":"yes","a2":"yes","a3":"yes","a4":"no","a5":"yes","a6":"no","a7":"yes"}
    vote(votes); summary()