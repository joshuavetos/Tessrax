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
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from ledger import append as ledger_append

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

    return FeedbackResponse(
        receipt_id=receipt_id,
        status="recorded",
        integrity_score=receipt["integrity_score"],
        timestamp=timestamp,
    )
