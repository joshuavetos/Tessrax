# wrp_v2_secure.py
# World Receipt Protocol v2 – Local Secure Build
# Author: Joshua Vetos / OpenAI GPT-5
# License: CC BY 4.0

"""
Run:  pip install fastapi uvicorn sqlalchemy slowapi python-jose
Then: python wrp_v2_secure.py
Docs: http://127.0.0.1:8000/docs
"""

import json, hashlib, datetime, math, time, uuid
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from slowapi import Limiter
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SECRET_KEY = "demo_secret_key_123"
ALGORITHM = "HS256"
DATABASE_URL = "sqlite:///wrp_v2.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="World Receipt Protocol v2 (Local Secure Build)")
app.state.limiter = limiter

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Actor(Base):
    __tablename__ = "actors"
    id = Column(String, primary_key=True)
    role = Column(String)   # user/org/admin
    email = Column(String, unique=True)
    bond_balance = Column(Float, default=10.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    receipts = relationship("Receipt", back_populates="actor")

class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(String, primary_key=True)
    actor_id = Column(String, ForeignKey("actors.id"))
    artifact_hash = Column(String)
    metadata = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    actor = relationship("Actor", back_populates="receipts")

class Challenge(Base):
    __tablename__ = "challenges"
    id = Column(String, primary_key=True)
    receipt_id = Column(String, ForeignKey("receipts.id"))
    challenger_id = Column(String, ForeignKey("actors.id"))
    evidence_hash = Column(String)
    status = Column(String, default="open")  # open/upheld/dismissed
    resolution = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

Base.metadata.create_all(engine)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

security = HTTPBearer()

def create_token(actor: Actor):
    payload = {
        "sub": actor.id,
        "email": actor.email,
        "role": actor.role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_current_actor(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        actor_id = payload.get("sub")
        db = SessionLocal()
        actor = db.query(Actor).filter_by(id=actor_id).first()
        db.close()
        if not actor:
            raise HTTPException(401, "Invalid actor")
        return actor
    except JWTError:
        raise HTTPException(401, "Invalid token")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ReceiptIn(BaseModel):
    artifact_hash: str
    content_type: str
    metadata: Dict[str, Any]

class ChallengeIn(BaseModel):
    receipt_id: str
    evidence_hash: str

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def dynamic_bond_required(content_type: str) -> float:
    multipliers = {
        "text": 1.0,
        "image": 2.0,
        "video": 5.0,
        "medical_advice": 10.0,
        "financial_prediction": 20.0,
    }
    return multipliers.get(content_type, 1.0)

def reputation_score(db, actor_id: str) -> float:
    chals = db.query(Challenge).filter_by(challenger_id=actor_id).all()
    score = 100.0
    now = datetime.datetime.utcnow()
    for c in chals:
        if c.status == "upheld" and c.resolved_at:
            days_ago = (now - c.resolved_at).days
            decay = math.exp(-days_ago / 365)
            score -= 10 * decay
    return max(0.0, min(100.0, score))

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "World Receipt Protocol v2 running", "docs": "/docs"}

# -- Auth -----------------------------------------------------

@app.post("/auth/login")
def demo_login(email: str):
    db = SessionLocal()
    actor = db.query(Actor).filter_by(email=email).first()
    if not actor:
        raise HTTPException(404, "Actor not found")
    token = create_token(actor)
    db.close()
    return {"access_token": token, "actor_id": actor.id, "role": actor.role}

# -- Register Receipt -----------------------------------------

@app.post("/register_receipt")
@limiter.limit("10/minute")
def register_receipt(r: ReceiptIn, actor: Actor = Depends(get_current_actor)):
    db = SessionLocal()
    required_bond = dynamic_bond_required(r.content_type)
    if actor.bond_balance < required_bond:
        db.close()
        raise HTTPException(402, f"Bond {required_bond} required")
    actor.bond_balance -= required_bond
    entry = Receipt(
        id=str(uuid.uuid4()),
        actor_id=actor.id,
        artifact_hash=r.artifact_hash,
        metadata=json.dumps(r.metadata, sort_keys=True)
    )
    db.add(entry)
    db.commit()
    db.close()
    return {"status": "ok", "receipt_id": entry.id, "bond_spent": required_bond}

# -- Verify Receipt -------------------------------------------

@app.get("/verify/{receipt_id}")
def verify_receipt(receipt_id: str):
    db = SessionLocal()
    r = db.query(Receipt).filter_by(id=receipt_id).first()
    db.close()
    if not r:
        raise HTTPException(404, "Not found")
    return {
        "id": r.id,
        "actor_id": r.actor_id,
        "artifact_hash": r.artifact_hash,
        "metadata": json.loads(r.metadata),
        "timestamp": r.timestamp
    }

# -- Challenge -------------------------------------------------

@app.post("/challenge")
def open_challenge(c: ChallengeIn, actor: Actor = Depends(get_current_actor)):
    db = SessionLocal()
    receipt = db.query(Receipt).filter_by(id=c.receipt_id).first()
    if not receipt:
        db.close()
        raise HTTPException(404, "Receipt not found")
    if actor.bond_balance < 1.0:
        db.close()
        raise HTTPException(402, "1.0 bond required to challenge")
    actor.bond_balance -= 1.0
    challenge = Challenge(
        id=str(uuid.uuid4()),
        receipt_id=receipt.id,
        challenger_id=actor.id,
        evidence_hash=c.evidence_hash
    )
    db.add(challenge)
    db.commit()
    db.close()
    return {"challenge_id": challenge.id, "status": "open"}

@app.post("/challenge/{cid}/resolve")
def resolve_challenge(cid: str, decision: str, actor: Actor = Depends(get_current_actor)):
    if actor.role != "admin":
        raise HTTPException(403, "Admin only")
    db = SessionLocal()
    ch = db.query(Challenge).filter_by(id=cid).first()
    if not ch:
        db.close()
        raise HTTPException(404, "Not found")
    if ch.status != "open":
        db.close()
        raise HTTPException(400, "Already resolved")

    ch.status = decision
    ch.resolved_at = datetime.datetime.utcnow()
    if decision == "upheld":
        # Slash receipt actor, reward challenger
        rec = db.query(Receipt).filter_by(id=ch.receipt_id).first()
        rec_actor = db.query(Actor).filter_by(id=rec.actor_id).first()
        challenger = db.query(Actor).filter_by(id=ch.challenger_id).first()
        rec_actor.bond_balance = max(0.0, rec_actor.bond_balance - 1.0)
        challenger.bond_balance += 1.0
    db.commit()
    db.close()
    return {"challenge_id": cid, "decision": decision}

# -- Reputation -----------------------------------------------

@app.get("/reputation/{actor_id}")
def get_reputation(actor_id: str):
    db = SessionLocal()
    score = reputation_score(db, actor_id)
    balance = db.query(Actor).filter_by(id=actor_id).first().bond_balance
    db.close()
    return {"actor_id": actor_id, "score": round(score, 2), "bond_balance": balance}

# -- Anchor Batch ---------------------------------------------

@app.post("/anchor_batch")
def anchor_batch():
    db = SessionLocal()
    receipts = db.query(Receipt.id, Receipt.artifact_hash).all()
    db.close()
    if not receipts:
        raise HTTPException(400, "No receipts")
    leaf_hashes = [hashlib.sha256(json.dumps([r[0], r[1]]).encode()).hexdigest()
                   for r in receipts]
    # Simple Merkle root (pairwise hash reduction)
    while len(leaf_hashes) > 1:
        if len(leaf_hashes) % 2 != 0:
            leaf_hashes.append(leaf_hashes[-1])
        leaf_hashes = [
            hashlib.sha256((leaf_hashes[i] + leaf_hashes[i+1]).encode()).hexdigest()
            for i in range(0, len(leaf_hashes), 2)
        ]
    return {"merkle_root": leaf_hashes[0], "anchored": len(receipts)}

# -- Demo Populate --------------------------------------------

@app.post("/demo/populate")
def populate_demo():
    db = SessionLocal()
    if db.query(Actor).count() > 0:
        db.close()
        return {"status": "already_populated"}

    actors = [
        Actor(id="user_alice", email="user_alice@example.com", role="user"),
        Actor(id="user_bob", email="user_bob@example.com", role="user"),
        Actor(id="org_reuters", email="org_reuters@example.com", role="org"),
        Actor(id="org_tessrax", email="org_tessrax@example.com", role="org"),
        Actor(id="admin", email="admin@example.com", role="admin", bond_balance=100.0)
    ]
    db.add_all(actors)
    db.commit()

    # create receipts
    for i in range(5):
        r = Receipt(
            id=str(uuid.uuid4()),
            actor_id="org_tessrax",
            artifact_hash=f"hash_{i}",
            metadata=json.dumps({"content_type": "text", "demo": i})
        )
        db.add(r)
    db.commit()

    # Add two challenges
    receipts = db.query(Receipt).all()
    c1 = Challenge(id=str(uuid.uuid4()), receipt_id=receipts[0].id,
                   challenger_id="user_bob", evidence_hash="ev_123", status="upheld",
                   resolved_at=datetime.datetime.utcnow())
    c2 = Challenge(id=str(uuid.uuid4()), receipt_id=receipts[1].id,
                   challenger_id="user_bob", evidence_hash="ev_456", status="dismissed",
                   resolved_at=datetime.datetime.utcnow())
    db.add_all([c1, c2])
    db.commit()
    db.close()
    return {"status": "ok", "actors": len(actors), "receipts": 5, "challenges": 2}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("\n✓ WRP v2 Secure Build ready — http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)