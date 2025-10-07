# wrp_v2.py
# World Receipt Protocol – v2 (Local Simulation Build)
# Adds: bonding, challenge system, Merkle anchoring, identity stubs
# Author: Joshua Vetos / Tessrax Stack
# License: CC BY 4.0

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import sqlite3, json, hashlib, os, uuid, random

# ---------------------------------------------------------------------
#  SETUP
# ---------------------------------------------------------------------
DB_PATH = "data/wrp_v2.db"
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.executescript("""
CREATE TABLE IF NOT EXISTS receipts(
    id TEXT PRIMARY KEY,
    artifact_hash TEXT,
    actor_id TEXT,
    model TEXT,
    signature TEXT,
    timestamp TEXT,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS bonds(
    actor_id TEXT PRIMARY KEY,
    balance REAL
);
CREATE TABLE IF NOT EXISTS challenges(
    id TEXT PRIMARY KEY,
    receipt_id TEXT,
    challenger_id TEXT,
    evidence_hash TEXT,
    status TEXT,
    opened_at TEXT,
    resolved_at TEXT,
    resolution TEXT
);
CREATE TABLE IF NOT EXISTS merkle_batches(
    id TEXT PRIMARY KEY,
    root_hash TEXT,
    receipt_count INTEGER,
    anchored_at TEXT
);
""")
conn.commit()

app = FastAPI(title="World Receipt Protocol v2 (Local Simulation)")

# ---------------------------------------------------------------------
#  MODELS
# ---------------------------------------------------------------------
class Receipt(BaseModel):
    artifact_hash: str
    actor_id: str
    model: str
    signature: str
    metadata: dict

class Bond(BaseModel):
    actor_id: str
    amount: float

class Challenge(BaseModel):
    receipt_id: str
    challenger_id: str
    evidence_hash: str

# ---------------------------------------------------------------------
#  UTILITIES
# ---------------------------------------------------------------------
def now(): return datetime.utcnow().isoformat()

def hash_dict(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def verify_actor_identity(actor_id: str) -> bool:
    """Stubbed identity check."""
    return actor_id.startswith("user_") or actor_id.startswith("org_")

# ---------------------------------------------------------------------
#  ENDPOINTS: IDENTITY & BONDING
# ---------------------------------------------------------------------
@app.post("/register_actor")
def register_actor(actor_id: str):
    """Register new actor with zero bonded balance."""
    if not verify_actor_identity(actor_id):
        raise HTTPException(403, "Invalid actor ID format.")
    c.execute("INSERT OR IGNORE INTO bonds(actor_id,balance) VALUES (?,0.0)", (actor_id,))
    conn.commit()
    return {"actor_id": actor_id, "registered": True}

@app.post("/bond/deposit")
def deposit_bond(b: Bond):
    """Deposit simulation funds (adds to bonded balance)."""
    if not verify_actor_identity(b.actor_id):
        raise HTTPException(403, "Unverified actor.")
    c.execute("INSERT OR IGNORE INTO bonds(actor_id,balance) VALUES (?,0.0)", (b.actor_id,))
    c.execute("UPDATE bonds SET balance=balance+? WHERE actor_id=?", (b.amount, b.actor_id))
    conn.commit()
    return {"actor_id": b.actor_id, "new_balance": c.execute("SELECT balance FROM bonds WHERE actor_id=?",(b.actor_id,)).fetchone()[0]}

# ---------------------------------------------------------------------
#  ENDPOINTS: RECEIPTS
# ---------------------------------------------------------------------
@app.post("/register_receipt")
def register_receipt(r: Receipt):
    """Register new signed receipt with actor’s bond check."""
    if not verify_actor_identity(r.actor_id):
        raise HTTPException(403, "Invalid actor.")
    balance = c.execute("SELECT balance FROM bonds WHERE actor_id=?",(r.actor_id,)).fetchone()
    if not balance or balance[0] < 1.0:
        raise HTTPException(402, "Insufficient bond (≥1.0 required).")
    rid = str(uuid.uuid4())
    c.execute("INSERT INTO receipts VALUES (?,?,?,?,?,?)",
              (rid, r.artifact_hash, r.actor_id, r.model, r.signature, now(), json.dumps(r.metadata)))
    conn.commit()
    return {"receipt_id": rid, "status": "recorded", "actor_balance": balance[0]}

@app.get("/verify/{artifact_hash}")
def verify_receipt(artifact_hash: str):
    row = c.execute("SELECT id,actor_id,model,signature,timestamp FROM receipts WHERE artifact_hash=?",(artifact_hash,)).fetchone()
    if not row: raise HTTPException(404, "No receipt found.")
    rid, actor, model, sig, ts = row
    return {"valid": True, "receipt_id": rid, "actor": actor, "model": model, "timestamp": ts}

# ---------------------------------------------------------------------
#  ENDPOINTS: CHALLENGES
# ---------------------------------------------------------------------
@app.post("/challenge")
def open_challenge(ch: Challenge):
    """Submit a contradiction or fraud challenge."""
    if not verify_actor_identity(ch.challenger_id):
        raise HTTPException(403, "Unverified challenger.")
    rid = ch.receipt_id
    if not c.execute("SELECT id FROM receipts WHERE id=?",(rid,)).fetchone():
        raise HTTPException(404, "Receipt not found.")
    cid = str(uuid.uuid4())
    c.execute("INSERT INTO challenges VALUES (?,?,?,?,?,?,?)",
              (cid, rid, ch.challenger_id, ch.evidence_hash, "open", now(), None, None))
    conn.commit()
    return {"challenge_id": cid, "status": "open"}

@app.post("/challenge/resolve/{challenge_id}")
def resolve_challenge(challenge_id: str, decision: str):
    """Resolve challenge manually (simulate arbitration)."""
    if decision not in ["upheld","dismissed"]:
        raise HTTPException(400, "Decision must be 'upheld' or 'dismissed'.")
    ch = c.execute("SELECT receipt_id,challenger_id FROM challenges WHERE id=?",(challenge_id,)).fetchone()
    if not ch: raise HTTPException(404, "Challenge not found.")
    rid, challenger = ch
    c.execute("UPDATE challenges SET status=?,resolved_at=?,resolution=? WHERE id=?",
              ("closed", now(), decision, challenge_id))
    # apply penalties
    receipt_actor = c.execute("SELECT actor_id FROM receipts WHERE id=?",(rid,)).fetchone()[0]
    if decision == "upheld":
        c.execute("UPDATE bonds SET balance=balance-1.0 WHERE actor_id=?",(receipt_actor,))
        c.execute("UPDATE bonds SET balance=balance+1.0 WHERE actor_id=?",(challenger,))
    else:
        c.execute("UPDATE bonds SET balance=balance-1.0 WHERE actor_id=?",(challenger,))
    conn.commit()
    return {"challenge_id": challenge_id, "decision": decision}

# ---------------------------------------------------------------------
#  ENDPOINTS: MERKLE BATCH ANCHORING (LOCAL)
# ---------------------------------------------------------------------
@app.post("/anchor_batch")
def anchor_batch():
    """Simulate Merkle batch anchoring of current receipts."""
    rows = c.execute("SELECT id,artifact_hash FROM receipts").fetchall()
    if not rows: raise HTTPException(400, "No receipts to anchor.")
    leaf_hashes = [hashlib.sha256((r[0]+r[1]).encode()).hexdigest() for r in rows]
    while len(leaf_hashes) > 1:
        if len(leaf_hashes)%2==1: leaf_hashes.append(leaf_hashes[-1])
        leaf_hashes = [hashlib.sha256((leaf_hashes[i]+leaf_hashes[i+1]).encode()).hexdigest()
                       for i in range(0,len(leaf_hashes),2)]
    root_hash = leaf_hashes[0]
    batch_id = str(uuid.uuid4())
    c.execute("INSERT INTO merkle_batches VALUES (?,?,?,?)",
              (batch_id, root_hash, len(rows), now()))
    conn.commit()
    return {"batch_id": batch_id, "root_hash": root_hash, "anchored": True}

@app.get("/anchors")
def list_anchors():
    rows = c.execute("SELECT id,root_hash,receipt_count,anchored_at FROM merkle_batches ORDER BY anchored_at DESC").fetchall()
    return [{"batch_id":r[0],"root":r[1],"count":r[2],"anchored_at":r[3]} for r in rows]

# ---------------------------------------------------------------------
#  ENDPOINTS: REPUTATION
# ---------------------------------------------------------------------
@app.get("/reputation/{actor_id}")
def reputation(actor_id: str):
    total = c.execute("SELECT COUNT(*) FROM receipts WHERE actor_id=?",(actor_id,)).fetchone()[0]
    upheld = c.execute("""
        SELECT COUNT(*) FROM challenges 
        WHERE receipt_id IN (SELECT id FROM receipts WHERE actor_id=?)
        AND resolution='upheld'
    """,(actor_id,)).fetchone()[0]
    score = max(0, 100 - (upheld/(total+0.001)*100))
    bal = c.execute("SELECT balance FROM bonds WHERE actor_id=?",(actor_id,)).fetchone()
    bal = bal[0] if bal else 0.0
    return {"actor_id": actor_id, "receipts": total, "upheld_challenges": upheld,
            "trust_score": round(score,2), "bond_balance": bal}

# ---------------------------------------------------------------------
#  DEMO INITIALIZER
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting WRP v2 (local) server at http://127.0.0.1:8000")
    uvicorn.run("wrp_v2:app", reload=True)