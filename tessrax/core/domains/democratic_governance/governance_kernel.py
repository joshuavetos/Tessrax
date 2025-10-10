"""
Governance kernel: propose and vote on balance mechanisms.
"""
from ledger import add_event, merkle_root
from quorum import check_quorum
from dissent import record_dissent

def propose_balance(name, proposer, rationale):
    add_event({"action":"propose","policy":name,"proposer":proposer,"rationale":rationale})
    print(f"Proposed: {name}")

def vote(votes):
    res=check_quorum(votes)
    add_event({"action":"vote","result":res,"votes":votes})
    if not res: record_dissent("democracy","Quorum failed")
    print("Vote:", "Accepted" if res else "Rejected")

def summary():
    print("Merkle Root:", merkle_root())

if __name__=="__main__":
    propose_balance("Participatory Delay Buffer","civic-lab",
                    "Introduce minimum deliberation window to protect representation.")
    votes={"a1":"yes","a2":"no","a3":"yes","a4":"yes","a5":"yes","a6":"yes","a7":"no"}
    vote(votes); summary()