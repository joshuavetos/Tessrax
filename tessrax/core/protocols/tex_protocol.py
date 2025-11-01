"""Inter-agent negotiation protocol schema for Tessrax agents.

This module fulfils AEP-001, RVC-001, and POST-AUDIT-001 by providing
deterministic schema validation with stub signing operations.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from tessrax.core.governance.receipts import write_receipt

PROTOCOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["claim", "evidence", "confidence", "truth_status", "next_step"],
    "properties": {
        "claim": {"type": "string"},
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "truth_status": {"type": "string"},
        "next_step": {"type": "string"},
    },
}


def sign_message(message: dict[str, Any]) -> str:
    """Produce deterministic signature stub for protocol messages."""

    serialized = "|".join(
        f"{key}:{message[key]}" for key in sorted(PROTOCOL_SCHEMA["required"])
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def verify_signature(message: dict[str, Any], signature: str) -> bool:
    """Verify stub signature by recomputing deterministic digest."""

    expected = sign_message(message)
    is_valid = expected == signature
    metrics = {"valid": 1 if is_valid else 0}
    write_receipt(
        "tessrax.core.protocols.tex_protocol.verify", "verified", metrics, 0.95
    )
    return is_valid


def validate_message(message: dict[str, Any]) -> bool:
    """Validate message against schema ranges and required fields."""

    for field in PROTOCOL_SCHEMA["required"]:
        if field not in message:
            return False
    confidence = message.get("confidence")
    if not isinstance(confidence, (int, float)):
        return False
    if confidence < 0.0 or confidence > 1.0:
        return False
    return True


def _self_test() -> bool:
    """Confirm schema validation and signature stubs operate correctly."""

    message = {
        "claim": "Solar output nominal",
        "evidence": "Meter reading 5kW",
        "confidence": 0.92,
        "truth_status": "probable",
        "next_step": "log",
    }
    assert validate_message(message), "Schema validation failed"
    signature = sign_message(message)
    assert verify_signature(message, signature), "Signature verification failed"
    write_receipt(
        "tessrax.core.protocols.tex_protocol.self_test",
        "verified",
        {"signature": signature[:12]},
        0.95,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
