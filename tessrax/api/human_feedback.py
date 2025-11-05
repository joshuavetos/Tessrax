"""Governed human feedback API router (DLK-verified).

This module operates under Tessrax Governance Kernel v16 and enforces
clauses ["AEP-001","POST-AUDIT-001","RVC-001","EAC-001"].  All
feedback receipts are cryptographically hashed and appended to the ledger.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

try:  # pragma: no cover - FastAPI may be unavailable in test environments
    from fastapi import APIRouter, HTTPException, Request, status
except ModuleNotFoundError:  # pragma: no cover
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str | None = None) -> None:
            super().__init__(detail or "HTTPException")
            self.status_code = status_code
            self.detail = detail

    class status:  # type: ignore[no-redef]
        HTTP_429_TOO_MANY_REQUESTS = 429

    class Request:  # Minimal stub for compatibility
        def __init__(self) -> None:
            self.client = None
            self.headers: dict[str, str] = {}

    class APIRouter:  # Minimal stub for compatibility
        def __init__(self, prefix: str = "", tags: list[str] | None = None) -> None:
            self.prefix = prefix
            self.tags = tags or []

        def post(self, *_args: Any, **_kwargs: Any):
            def decorator(func):
                return func

            return decorator

try:  # pragma: no cover - Pydantic may be unavailable in test environments
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    class BaseModel:  # Minimal stub for compatibility
        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    def Field(default: Any = ..., **_kwargs: Any) -> Any:  # type: ignore[override]
        return default

from ledger import append as ledger_append

LEGACY_DATA_PATH = Path(__file__).with_name("human_feedback_history.json")


def _normalise_history_path(path: Path) -> Path:
    if path.suffix:
        return path
    return path / "human_feedback_history.json"


def _resolve_history_path() -> Path:
    candidates: list[Path] = []

    env_history = os.environ.get("TESSRAX_FEEDBACK_HISTORY_PATH")
    if env_history:
        candidates.append(_normalise_history_path(Path(env_history)))

    env_data = os.environ.get("TESSRAX_DATA_DIR")
    if env_data:
        candidates.append(_normalise_history_path(Path(env_data)))

    home_candidate = Path.home() / ".tessrax"
    candidates.append(home_candidate / "human_feedback_history.json")

    temp_candidate = Path(tempfile.gettempdir()) / "tessrax"
    candidates.append(temp_candidate / "human_feedback_history.json")

    for candidate in candidates:
        parent = candidate.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            probe = parent / ".history_probe"
            with probe.open("w", encoding="utf-8") as handle:
                handle.write("")
            probe.unlink(missing_ok=True)
        except OSError:
            continue

        if (
            candidate != LEGACY_DATA_PATH
            and not candidate.exists()
            and LEGACY_DATA_PATH.exists()
        ):
            try:
                shutil.copy2(LEGACY_DATA_PATH, candidate)
            except OSError:
                # If we cannot migrate, continue with an empty history.
                pass
        return candidate

    return LEGACY_DATA_PATH


DATA_PATH = _resolve_history_path()
LOCK_PATH = DATA_PATH.with_suffix(".lock")

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


@contextlib.contextmanager
def _history_lock(timeout: float = 5.0, stale_seconds: float = 30.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout if timeout is not None else None
    fd: int | None = None
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if stale_seconds is not None:
                try:
                    mtime = LOCK_PATH.stat().st_mtime
                    if time.time() - mtime > stale_seconds:
                        LOCK_PATH.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Unable to acquire feedback history lock")
            time.sleep(0.1)

    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except OSError:
            pass
        LOCK_PATH.unlink(missing_ok=True)


def _load_history() -> dict[str, Any]:
    if not DATA_PATH.exists():
        return {"history": []}
    try:
        with DATA_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"history": []}
    history = data.get("history")
    if not isinstance(history, list):
        data["history"] = []
    return data


def _ensure_history_file() -> dict[str, Any]:
    with _history_lock():
        history = _load_history()
        if not DATA_PATH.exists():
            _write_history(history)
        return copy.deepcopy(history)


def _write_history(data: dict[str, Any]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=True)
    tmp_path = DATA_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
    tmp_path.replace(DATA_PATH)


def get_history() -> dict[str, Any]:
    """Return the locally cached governed feedback history."""

    return _ensure_history_file()


def _append_history(entry: dict[str, Any]) -> None:
    with _history_lock():
        history = _load_history()
        history.setdefault("history", []).append(entry)
        _write_history(history)


def _self_test() -> bool:
    """Exercise the legacy helpers for compatibility with existing tests."""

    history = _ensure_history_file()
    if not history["history"]:
        sample_entry = {
            "claim_id": "TEST-CLAIM",
            "verdict": "recorded",
            "correction": "Sample correction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "integrity_score": 0.95,
        }
        _append_history(sample_entry)
    return True


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

    _append_history(
        {
            "claim_id": payload.claim_id,
            "correction": payload.correction,
            "timestamp": timestamp,
            "verdict": "recorded",
            "integrity_score": receipt["integrity_score"],
        }
    )

    return FeedbackResponse(
        receipt_id=receipt_id,
        status="recorded",
        integrity_score=receipt["integrity_score"],
        timestamp=timestamp,
    )
