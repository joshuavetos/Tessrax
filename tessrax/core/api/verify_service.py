"""
Tessrax Verification API
Public endpoint for verifying Merkle roots and receipts.
Run with: uvicorn core.api.verify_service:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.ledger.merkle_api import build_ledger_cache, merkle_root, get_proof, verify_proof
from core.anchor.verify_anchor import ANCHOR_LOG
import json, time, hashlib, os

app = FastAPI(title="Tessrax Verification API", version="1.0")

class Receipt(BaseModel):
    source_hash: str
    contradiction_score: float
    timestamp: int
    signature: str | None = None
    domain: str | None = None

@app.get("/")
def root():
    return {
        "service": "Tessrax Verification API",
        "description": "Public verifier for Merkle roots and anchored receipts",
        "status": "online"
    }

@app.get("/verify/root/{root_hash}")
def verify_root(root_hash:str):
    """Check if root exists in the anchor log."""
    if not os.path.exists(ANCHOR_LOG):
        raise HTTPException(404, "Anchor log not found.")
    with open(ANCHOR_LOG) as f:
        for line in f:
            rec=json.loads(line)
            if rec["root"]==root_hash:
                ts=time.ctime(rec["timestamp"])
                return {
                    "root": root_hash,
                    "anchored": True,
                    "timestamp": ts,
                    "proofs": rec["proofs"]
                }
    raise HTTPException(404, "Root not found in anchor log.")

@app.post("/verify/receipt")
def verify_receipt(receipt:Receipt):
    """Verify that a specific receipt is in the ledger and covered by an anchored root."""
    ledger=build_ledger_cache()
    root=merkle_root(ledger)
    idx=None
    for i,rec in enumerate(ledger):
        if rec.get("source_hash")==receipt.source_hash:
            idx=i; break
    if idx is None:
        raise HTTPException(404, "Receipt not found in ledger.")
    proof=get_proof(ledger,idx)
    ok=verify_proof(ledger[idx],proof,root)
    return {
        "source_hash": receipt.source_hash,
        "included": ok,
        "root": root,
        "proof_path": proof,
        "anchored": root_in_log(root)
    }

def root_in_log(root:str)->bool:
    if not os.path.exists(ANCHOR_LOG): return False
    with open(ANCHOR_LOG) as f:
        for line in f:
            rec=json.loads(line)
            if rec["root"]==root:
                return True
    return False