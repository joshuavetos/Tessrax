# WorldReceiptProtocol Prototype v1.0
# Author: Joshua Vetos / Tessrax LLC
# License: CC BY 4.0
# Dependencies: fastapi, uvicorn, pydantic, cryptography, sqlite3 (stdlib)

import json, sqlite3, hashlib, datetime, uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from typing import Optional

DB_PATH = "world_receipts.db"

# ---------------------------------------------------------------------
# DB Init
# ---------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
conn.execute("""
CREATE TABLE IF NOT EXISTS receipts (
    id TEXT PRIMARY KEY,
    artifact_hash TEXT NOT NULL,
    artifact_type TEXT,
    provenance_chain TEXT,
    actor_id TEXT,
    role TEXT,
    signature TEXT,
    parent_receipt TEXT,
    timestamp TEXT
)
""")
conn.commit()

# ---------------------------------------------------------------------
# Key Management (demo only)
# ---------------------------------------------------------------------
PRIVATE_KEY = ed25519.Ed25519PrivateKey.generate()
PUBLIC_KEY = PRIVATE_KEY.public_key()

def sign_payload(data: dict) -> str:
    """Create an Ed25519 signature for canonical JSON."""
    canonical = json.dumps(data, sort_keys=True).encode()
    return PRIVATE_KEY.sign(canonical).hex()

def verify_signature(data: dict, signature_hex: str) -> bool:
    canonical = json.dumps(data, sort_keys=True).encode()
    try:
        PUBLIC_KEY.verify(bytes.fromhex(signature_hex), canonical)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class ReceiptIn(BaseModel):
    artifact_type: str
    artifact_hash: str
    actor_id: str
    role: str = "creator"
    provenance_chain: Optional[list[str]] = []
    parent_receipt: Optional[str] = None

class ReceiptOut(BaseModel):
    receipt_id: str
    timestamp: str
    artifact_type: str
    artifact_hash: str
    actor_id: str
    role: str
    provenance_chain: list[str]
    parent_receipt: Optional[str]
    signature: str

# ---------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------
app = FastAPI(title="World Receipt Protocol", version="1.0")

@app.post("/register", response_model=ReceiptOut)
def register_receipt(r: ReceiptIn):
    """Register a new receipt (hash -> signature -> store)."""
    receipt_id = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()

    payload = {
        "receipt_id": receipt_id,
        "timestamp": timestamp,
        "artifact_type": r.artifact_type,
        "artifact_hash": r.artifact_hash,
        "actor_id": r.actor_id,
        "role": r.role,
        "provenance_chain": r.provenance_chain,
        "parent_receipt": r.parent_receipt
    }

    signature = sign_payload(payload)

    with conn:
        conn.execute("""
        INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            receipt_id,
            r.artifact_hash,
            r.artifact_type,
            json.dumps(r.provenance_chain),
            r.actor_id,
            r.role,
            signature,
            r.parent_receipt,
            timestamp
        ))

    return {**payload, "signature": signature}

@app.get("/verify/{artifact_hash}")
def verify_receipt(artifact_hash: str):
    """Verify receipt integrity and signature by artifact hash."""
    row = conn.execute("""
        SELECT * FROM receipts WHERE artifact_hash=?
    """, (artifact_hash,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Receipt not found")

    (
        receipt_id, a_hash, a_type, chain_json,
        actor_id, role, signature, parent, ts
    ) = row

    payload = {
        "receipt_id": receipt_id,
        "timestamp": ts,
        "artifact_type": a_type,
        "artifact_hash": a_hash,
        "actor_id": actor_id,
        "role": role,
        "provenance_chain": json.loads(chain_json),
        "parent_receipt": parent
    }

    valid = verify_signature(payload, signature)
    return {"valid": valid, "payload": payload, "signature": signature}

@app.get("/public_key")
def get_public_key():
    """Expose the demo public key for verification."""
    pub_hex = PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    ).hex()
    return {"public_key": pub_hex}

# ---------------------------------------------------------------------
# Utility (local CLI test)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("✓ WorldReceiptProtocol running on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)