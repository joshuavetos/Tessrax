# wrp_v2_1_hardlock.py
# --------------------------------------------------------------
# WorldReceiptProtocol v2.1 "Hardlock"
# Hardened local-only prototype of a verifiable provenance ledger
# Author: Joshua Vetos / GPT-5 (Tessrax)
# --------------------------------------------------------------

import uuid
import json
import time
import hashlib
import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from collections import defaultdict

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, String, Float, Text,
    DateTime, ForeignKey, Integer
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# --------------------------------------------------------------
# Database setup (SQLite local)
# --------------------------------------------------------------
engine = create_engine("sqlite:///wrp_local.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------
# Auth placeholder (local dev)
# --------------------------------------------------------------
security = HTTPBearer()
TOKENS = {"user_admin": "admin", "user_alpha": "user", "user_beta": "user"}


def get_current_actor(token: HTTPAuthorizationCredentials = Depends(security)):
    actor_id = token.credentials
    if actor_id not in TOKENS:
        raise HTTPException(403, "Invalid token")
    return {"id": actor_id, "role": TOKENS[actor_id]}


# --------------------------------------------------------------
# ORM MODELS
# --------------------------------------------------------------
class Actor(Base):
    __tablename__ = "actors"
    id = Column(String, primary_key=True)
    role = Column(String, default="user")
    bond_balance = Column(Float, default=10.0)
    bond_lifetime_spent = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(String, primary_key=True)
    actor_id = Column(String, ForeignKey("actors.id"))
    artifact_hash = Column(String)
    content_type = Column(String)
    metadata = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class Challenge(Base):
    __tablename__ = "challenges"
    id = Column(String, primary_key=True)
    receipt_id = Column(String, ForeignKey("receipts.id"))
    challenger_id = Column(String, ForeignKey("actors.id"))
    evidence_hash = Column(String)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class Jury(Base):
    __tablename__ = "jury_pool"
    actor_id = Column(String, ForeignKey("actors.id"), primary_key=True)
    reputation_score = Column(Float, default=100.0)
    active = Column(Integer, default=1)
    total_votes = Column(Integer, default=0)
    correct_votes = Column(Integer, default=0)


class Vote(Base):
    __tablename__ = "votes"
    id = Column(String, primary_key=True)
    challenge_id = Column(String, ForeignKey("challenges.id"))
    voter_id = Column(String, ForeignKey("actors.id"))
    decision = Column(String)
    stake = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class MerkleBatch(Base):
    __tablename__ = "merkle_batches"
    id = Column(String, primary_key=True)
    root_hash = Column(String)
    tree_json = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class MerkleProof(Base):
    __tablename__ = "merkle_proofs"
    receipt_id = Column(String, ForeignKey("receipts.id"), primary_key=True)
    batch_id = Column(String, ForeignKey("merkle_batches.id"))
    proof_path = Column(Text)
    leaf_index = Column(Integer)


Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------------------
class ReceiptIn(BaseModel):
    artifact_hash: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = {}


class ChallengeIn(BaseModel):
    receipt_id: str
    evidence_hash: str


# --------------------------------------------------------------
# Utility / Config
# --------------------------------------------------------------
def dynamic_bond_required(content_type: str) -> float:
    return {
        "text": 1.0,
        "image": 2.0,
        "video": 5.0,
        "medical": 10.0,
        "financial": 20.0,
    }.get(content_type, 1.0)


actor_rate_limits = defaultdict(list)


def rate_limit_actor(actor_id: str, max_requests: int, window_seconds: int):
    now = time.time()
    actor_rate_limits[actor_id] = [
        ts for ts in actor_rate_limits[actor_id] if now - ts < window_seconds
    ]
    if len(actor_rate_limits[actor_id]) >= max_requests:
        raise HTTPException(429, f"Rate limit exceeded ({max_requests}/{window_seconds}s)")
    actor_rate_limits[actor_id].append(now)


# --------------------------------------------------------------
# APP INIT
# --------------------------------------------------------------
app = FastAPI(title="WorldReceiptProtocol v2.1 Hardlock")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------
# CORE ENDPOINTS
# --------------------------------------------------------------
@app.post("/bond/deposit")
def deposit_bond(amount: float, actor=Depends(get_current_actor)):
    with get_db() as db:
        obj = db.query(Actor).filter_by(id=actor["id"]).first()
        if not obj:
            obj = Actor(id=actor["id"], role=actor["role"])
            db.add(obj)
        obj.bond_balance += amount
        db.commit()
        return {"actor": actor["id"], "new_balance": obj.bond_balance}


@app.post("/register_receipt")
def register_receipt(r: ReceiptIn, actor=Depends(get_current_actor)):
    rate_limit_actor(actor["id"], 10, 60)
    with get_db() as db:
        a = db.query(Actor).filter_by(id=actor["id"]).first()
        if not a:
            raise HTTPException(403, "Actor not found, deposit bond first")

        req_bond = dynamic_bond_required(r.content_type)
        if a.bond_lifetime_spent >= 1000.0:
            raise HTTPException(429, "Lifetime bond limit reached")
        if a.bond_balance < req_bond:
            raise HTTPException(402, "Insufficient bond")

        a.bond_balance -= req_bond
        a.bond_lifetime_spent += req_bond

        rid = str(uuid.uuid4())
        rec = Receipt(
            id=rid,
            actor_id=actor["id"],
            artifact_hash=r.artifact_hash,
            content_type=r.content_type,
            metadata=json.dumps(r.metadata),
        )
        db.add(rec)
        db.commit()
        return {"receipt_id": rid, "actor": actor["id"], "bond_spent": req_bond}


@app.post("/challenge")
def open_challenge(ch: ChallengeIn, actor=Depends(get_current_actor)):
    with get_db() as db:
        target = db.query(Receipt).filter_by(id=ch.receipt_id).first()
        if not target:
            raise HTTPException(404, "Receipt not found")
        challenger = db.query(Actor).filter_by(id=actor["id"]).first()
        if challenger.bond_balance < 1.0:
            raise HTTPException(402, "1.0 bond required to challenge")
        challenger.bond_balance -= 1.0
        cid = str(uuid.uuid4())
        c = Challenge(
            id=cid,
            receipt_id=ch.receipt_id,
            challenger_id=actor["id"],
            evidence_hash=ch.evidence_hash,
        )
        db.add(c)
        db.commit()
        return {"challenge_id": cid, "locked_bond": 1.0}


@app.post("/jury/enroll")
def enroll_jury(actor=Depends(get_current_actor)):
    with get_db() as db:
        a = db.query(Actor).filter_by(id=actor["id"]).first()
        if not a or a.bond_balance < 10.0:
            raise HTTPException(402, "10.0 bond required")
        rep = 100.0
        db.add(Jury(actor_id=actor["id"], reputation_score=rep))
        db.commit()
        return {"jury_enrolled": actor["id"], "score": rep}


@app.post("/challenge/{cid}/vote")
def vote_challenge(cid: str, decision: str, stake: float, actor=Depends(get_current_actor)):
    with get_db() as db:
        jury = db.query(Jury).filter_by(actor_id=actor["id"], active=1).first()
        if not jury:
            raise HTTPException(403, "Not in jury pool")
        a = db.query(Actor).filter_by(id=actor["id"]).first()
        if a.bond_balance < stake:
            raise HTTPException(402, "Insufficient bond")
        a.bond_balance -= stake
        v = Vote(
            id=str(uuid.uuid4()),
            challenge_id=cid,
            voter_id=actor["id"],
            decision=decision,
            stake=stake,
        )
        db.add(v)
        db.commit()
        return {"vote_id": v.id, "stake": stake}


@app.post("/challenge/{cid}/finalize")
def finalize_challenge(cid: str):
    with get_db() as db:
        ch = db.query(Challenge).filter_by(id=cid).first()
        if not ch:
            raise HTTPException(404, "Challenge not found")
        votes = db.query(Vote).filter_by(challenge_id=cid).all()
        if not votes:
            raise HTTPException(400, "No votes")
        upheld = sum(v.stake for v in votes if v.decision == "upheld")
        dismissed = sum(v.stake for v in votes if v.decision == "dismissed")
        final_decision = "upheld" if upheld > dismissed else "dismissed"
        ch.status = final_decision
        ch.resolved_at = datetime.datetime.utcnow()
        total_pool = upheld + dismissed
        winning = upheld if final_decision == "upheld" else dismissed
        for v in votes:
            voter = db.query(Actor).filter_by(id=v.voter_id).first()
            jury = db.query(Jury).filter_by(actor_id=v.voter_id).first()
            if v.decision == final_decision:
                reward = v.stake + (v.stake / winning) * total_pool * 0.1
                voter.bond_balance += reward
                jury.correct_votes += 1
            jury.total_votes += 1
            jury.reputation_score = (jury.correct_votes / jury.total_votes) * 100
        db.commit()
        return {"challenge": cid, "decision": final_decision}


# --------------------------------------------------------------
# MERKLE BATCHING (local proof only)
# --------------------------------------------------------------
def build_merkle_tree_with_proofs(leaves: List[str]) -> Dict[str, Any]:
    import merkletools
    mt = merkletools.MerkleTools(hash_type="sha256")
    for leaf in leaves:
        mt.add_leaf(leaf, do_hash=False)
    mt.make_tree()
    proofs = {}
    for i, leaf in enumerate(leaves):
        proof = mt.get_proof(i)
        proofs[leaf] = {
            "leaf_index": i,
            "proof": proof,
        }
    return {"root": mt.get_merkle_root(), "tree": mt.get_tree(), "proofs": proofs}


@app.post("/anchor_batch")
def anchor_batch():
    with get_db() as db:
        receipts = db.query(Receipt.id, Receipt.artifact_hash).all()
        if not receipts:
            raise HTTPException(400, "No receipts")
        leaves = [
            hashlib.sha256(json.dumps([r.id, r.artifact_hash]).encode()).hexdigest()
            for r in receipts
        ]
        result = build_merkle_tree_with_proofs(leaves)
        batch_id = str(uuid.uuid4())
        batch = MerkleBatch(
            id=batch_id,
            root_hash=result["root"],
            tree_json=json.dumps(result["tree"]),
        )
        db.add(batch)
        for i, r in enumerate(receipts):
            leaf_hash = leaves[i]
            proof = MerkleProof(
                receipt_id=r.id,
                batch_id=batch_id,
                proof_path=json.dumps(result["proofs"][leaf_hash]["proof"]),
                leaf_index=i,
            )
            db.add(proof)
        db.commit()
        return {"batch": batch_id, "root": result["root"], "anchored": len(leaves)}


@app.get("/merkle_proof/{rid}")
def get_merkle_proof(rid: str):
    with get_db() as db:
        p = db.query(MerkleProof).filter_by(receipt_id=rid).first()
        if not p:
            raise HTTPException(404, "Not anchored")
        b = db.query(MerkleBatch).filter_by(id=p.batch_id).first()
        return {
            "receipt_id": rid,
            "batch": p.batch_id,
            "root_hash": b.root_hash,
            "proof": json.loads(p.proof_path),
            "leaf_index": p.leaf_index,
        }


# --------------------------------------------------------------
# HEALTH / UTILITY
# --------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "version": "2.1-hardlock"}


@app.get("/actor/{aid}")
def actor_info(aid: str):
    with get_db() as db:
        a = db.query(Actor).filter_by(id=aid).first()
        if not a:
            raise HTTPException(404)
        return {
            "actor": a.id,
            "role": a.role,
            "bond_balance": a.bond_balance,
            "lifetime_spent": a.bond_lifetime_spent,
        }