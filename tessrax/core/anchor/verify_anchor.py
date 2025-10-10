"""
Verifies that a Merkle root was publicly anchored.
"""

import json, time, requests
from core.ledger.merkle_api import merkle_root, build_ledger_cache

ANCHOR_LOG = "logs/anchor_log.jsonl"

def verify_anchor():
    ledger=build_ledger_cache()
    root=merkle_root(ledger)
    found=False
    with open(ANCHOR_LOG) as f:
        for line in f:
            rec=json.loads(line)
            if rec["root"]==root:
                found=True
                ts=time.ctime(rec["timestamp"])
                print(f"Root {root[:12]}… anchored on {ts}")
                for p in rec["proofs"]:
                    print(f"  via {p['method']}: {p.get('cid',p.get('proof'))}")
    if not found:
        print("Root not yet anchored.")

if __name__=="__main__":
    verify_anchor()