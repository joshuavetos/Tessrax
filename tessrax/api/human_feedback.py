"""Governed human feedback API router (DLK-verified).

This module operates under Tessrax Governance Kernel v16 and enforces
clauses ["AEP-001","POST-AUDIT-001","RVC-001","EAC-001"].  All
feedback receipts are cryptographically hashed and appended to the ledger.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ledger import append as ledger_append

DATA_PATH = Path("out/human_feedback_history.json")
_CORRUPT_SUFFIX = ".corrupted"
_SELF_TEST_TIMESTAMP = "1970-01-01T00:00:00+00:00"

AUDITOR_ID = "Tessrax Governance Kernel v16"
CLAUSES = ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"]


class FeedbackSubmission(BaseModel):
    """Pydantic schema for governed human feedback payloads."""

    claim_id: str = Field(..., min_length=1, description="Claim identifier")
    correction: str = Field(..., min_length=1, description="Suggested correction")
    user_id: str = Field(..., min_length=1, description="Internal pseudonymous user ID")
    signature: str | None = Field(
        default=None, description="Optional user-provided digital signature"
    )
    metadata: Dict[str, Any] | None = Field(
        default=None, description="Optional structured metadata"
    )


class FeedbackResponse(BaseModel):
    """Response payload returning the governed feedback receipt."""

    receipt_id: str
    status: str
    integrity_score: float
    timestamp: str


class _RateLimiter:
    """Simple in-memory rate limiter with monotonic time accounting."""

    def __init__(self, limit: int = 3, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window = window_seconds
        self._events: dict[str, list[float]] = {}

    def verify(self, key: str) -> None:
        now = time.monotonic()
        attempts = [t for t in self._events.get(key, []) if now - t < self.window]
        if len(attempts) >= self.limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Feedback rate limit exceeded; please retry later.",
            )
        attempts.append(now)
        self._events[key] = attempts


router = APIRouter(prefix="/feedback", tags=["Human Feedback"])
_rate_limiter = _RateLimiter()


def _hash_ip(ip: str) -> str:
    digest = hashlib.sha256(ip.encode("utf-8")).hexdigest()
    return digest[:16]


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackSubmission, request: Request) -> FeedbackResponse:
    """Persist governed human feedback and append a ledger receipt."""

    client = request.client
    ip = client.host if client else "0.0.0.0"
    user_agent = request.headers.get("user-agent", "unknown")
    key = f"{payload.user_id}:{_hash_ip(ip)}"
    _rate_limiter.verify(key)

    timestamp = datetime.now(timezone.utc).isoformat()
    receipt = {
        "auditor": AUDITOR_ID,
        "clauses": CLAUSES,
        "timestamp": timestamp,
        "event_type": "HUMAN_FEEDBACK_RECEIPT",
        "claim_id": payload.claim_id,
        "correction": payload.correction,
        "user_id": payload.user_id,
        "user_signature": payload.signature,
        "metadata": payload.metadata or {},
        "ip_hash": _hash_ip(ip),
        "user_agent": user_agent[:256],
        "status": "recorded",
        "integrity_score": 0.95,
    }
    serialized = json.dumps(receipt, sort_keys=True).encode("utf-8")
    receipt_id = hashlib.sha256(serialized).hexdigest()[:32]
    ledger_append({"event_type": "HUMAN_FEEDBACK_RECEIPT", "payload": receipt})

    history = get_history()
    history.setdefault("history", []).append({
        "receipt_id": receipt_id,
        "verdict": receipt["status"],
        "timestamp": timestamp,
    })
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return FeedbackResponse(
        receipt_id=receipt_id,
        status="recorded",
        integrity_score=receipt["integrity_score"],
        timestamp=timestamp,
    )


def get_history() -> dict[str, Any]:
    if DATA_PATH.exists():
        try:
            payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            history = payload.get("history", []) if isinstance(payload, dict) else []
            if not isinstance(history, list):
                raise ValueError("History payload must be a list")
            return {"history": history}
        except (json.JSONDecodeError, ValueError):
            quarantine_corrupt_history()
            return {"history": []}
    return {"history": []}


def quarantine_corrupt_history() -> None:
    """Move a corrupt history file aside so future writes succeed."""

    if not DATA_PATH.exists():
        return
    backup_path = DATA_PATH.with_suffix(DATA_PATH.suffix + _CORRUPT_SUFFIX)
    try:
        if backup_path.exists():
            backup_path.unlink()
        DATA_PATH.replace(backup_path)
    except OSError:
        DATA_PATH.unlink(missing_ok=True)


def _self_test() -> bool:
    probe = {
        "receipt_id": "self-test",
        "verdict": "verified",
        "timestamp": _SELF_TEST_TIMESTAMP,
    }
    history = get_history()
    existing = [entry for entry in history.get("history", []) if entry.get("receipt_id") != probe["receipt_id"]]
    existing.append(probe)
    history["history"] = existing
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    persisted = get_history()["history"]
    if not persisted or persisted[-1] != probe:
        raise RuntimeError("Human feedback self-test failed to persist probe entry")
    return True
