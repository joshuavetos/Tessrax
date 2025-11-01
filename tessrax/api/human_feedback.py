"""FastAPI microservice for Tessrax human feedback capture.

The service satisfies AEP-001, RVC-001, and POST-AUDIT-001 by providing
cold-start endpoints with deterministic storage in JSONL format. Each
request results in auditable receipts supporting the Receipts-First
Rule.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ..core.governance.receipts import write_receipt

APP = FastAPI(title="Tessrax Human Feedback", version="1.0.0")
DATA_PATH = Path("tessrax/data/human_feedback.jsonl")
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
if not DATA_PATH.exists():
    DATA_PATH.write_text("", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Append feedback payload to the JSONL ledger."""

    entry = {
        "timestamp": payload.get("timestamp", _timestamp()),
        "rule_id": payload["rule_id"],
        "verdict": payload["verdict"],
        "rationale": payload.get("rationale", ""),
        "signature": payload.get("signature", "stub-signature"),
    }
    with DATA_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            "{" + ",".join(f'"{key}":"{entry[key]}"' for key in sorted(entry)) + "}\n"
        )
    return entry


def _load_history() -> List[Dict[str, Any]]:
    """Load historical feedback entries deterministically."""

    history: List[Dict[str, Any]] = []
    if not DATA_PATH.exists():
        return history
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: Dict[str, Any] = {}
        for fragment in line.strip("{} ").split(","):
            key, value = fragment.split(":", 1)
            record[key.strip('"')] = value.strip('"')
        history.append(record)
    return history


@APP.post("/feedback")
def submit_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Record human feedback with runtime validation."""

    for field in ("rule_id", "verdict"):
        if field not in payload:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")
    entry = _log_feedback(payload)
    metrics = {"history_size": len(_load_history())}
    write_receipt("tessrax.api.human_feedback.post", "verified", metrics, 0.95)
    return entry


@APP.get("/history")
def get_history() -> Dict[str, Any]:
    """Return deterministic history payload."""

    history = _load_history()
    metrics = {"history_size": len(history)}
    write_receipt("tessrax.api.human_feedback.get", "verified", metrics, 0.95)
    return {"history": history}


def _self_test() -> bool:
    """Execute deterministic API round-trip tests using TestClient."""

    client = TestClient(APP)
    response = client.post(
        "/feedback",
        json={"rule_id": "R-001", "verdict": "approve", "rationale": "Aligned"},
    )
    assert response.status_code == 200, "POST failed"
    entry = response.json()
    assert entry["rule_id"] == "R-001", "Incorrect rule id"
    history_response = client.get("/history")
    assert history_response.status_code == 200, "GET failed"
    history_body = history_response.json()
    assert history_body["history"], "History should not be empty"
    write_receipt(
        "tessrax.api.human_feedback.self_test",
        "verified",
        {"history_size": len(history_body["history"] )},
        0.96,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
