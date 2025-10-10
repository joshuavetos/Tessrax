"""
Command-line verifier.
Usage:
    python -m core.api.verify_receipt <receipt.json>
"""

import json, sys
from core.ledger.merkle_api import build_ledger_cache, get_proof, merkle_root, verify_proof

def verify_single(receipt_path):
    with open(receipt_path) as f:
        receipt=json.load(f)
    ledger=build_ledger_cache()
    root=merkle_root(ledger)
    # find index
    idx=None
    for i,rec in enumerate(ledger):
        if rec.get("source_hash")==receipt.get("source_hash"):
            idx=i; break
    if idx is None:
        print("Receipt not found.")
        return
    proof=get_proof(ledger,idx)
    ok=verify_proof(ledger[idx],proof,root)
    print(f"Ledger Root: {root}")
    print(f"Receipt Index: {idx}")
    print(f"Verification Result: {'VALID' if ok else 'INVALID'}")

if __name__=="__main__":
    verify_single(sys.argv[1])