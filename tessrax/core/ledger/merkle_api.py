"""
Merkle Proof API for Tessrax Ledger
Allows external verification that a receipt exists in the ledger without
disclosing all other entries.

Dependencies: hashlib, json
"""

import hashlib, json, os
LEDGER_PATH = "ledger.jsonl"

def sha(x:str)->str:
    return hashlib.sha256(x.encode()).hexdigest()

def build_merkle_tree(entries:list)->list:
    """Return Merkle tree as list of levels; level[0] = leaf hashes."""
    level=[sha(json.dumps(e,sort_keys=True)) for e in entries]
    tree=[level]
    while len(level)>1:
        next_level=[]
        for i in range(0,len(level),2):
            left=level[i]
            right=level[i+1] if i+1<len(level) else left
            next_level.append(sha(left+right))
        tree.append(next_level)
        level=next_level
    return tree

def merkle_root(entries:list)->str:
    """Convenience wrapper."""
    return build_merkle_tree(entries)[-1][0]

def get_proof(entries:list,index:int)->list:
    """Return Merkle proof (hash path) for entry at index."""
    tree=build_merkle_tree(entries)
    proof=[]
    for level in tree[:-1]:
        pair_index=index^1  # sibling node
        sibling_hash=level[pair_index] if pair_index<len(level) else level[index]
        proof.append(sibling_hash)
        index//=2
    return proof

def verify_proof(entry,proof,root)->bool:
    """Recreate hash up the tree and check against root."""
    h=sha(json.dumps(entry,sort_keys=True))
    for sibling in proof:
        combined="".join(sorted([h,sibling]))
        h=sha(combined)
    return h==root

def build_ledger_cache():
    if not os.path.exists(LEDGER_PATH):
        print("Ledger not found.")
        return []
    with open(LEDGER_PATH) as f:
        return [json.loads(line) for line in f]

if __name__=="__main__":
    entries=build_ledger_cache()
    root=merkle_root(entries)
    idx=0
    proof=get_proof(entries,idx)
    print("Root:",root)
    print("Proof for entry 0:",proof)
    ok=verify_proof(entries[idx],proof,root)
    print("Verification:",ok)