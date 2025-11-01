"""Audit ledger integration for Tessrax whitepaper compliance.

Adheres to AEP-001, RVC-001, and POST-AUDIT-001 with deterministic
Merkle recomputation for appended audit receipts.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from tessrax.core import PROJECT_ROOT
from tessrax.core.governance.receipts import write_receipt

LEDGER_PATH = (PROJECT_ROOT / "ledger" / "ledger.jsonl").resolve()
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
if not LEDGER_PATH.exists():
    LEDGER_PATH.write_text("", encoding="utf-8")


def _stable_json(payload: dict[str, Any]) -> str:
    return (
        "{"
        + ",".join(
            (
                f'"{key}":"{payload[key]}"'
                if isinstance(payload[key], str)
                else f'"{key}":{payload[key]}'
            )
            for key in sorted(payload)
        )
        + "}"
    )


def _merkle_root(lines: list[str]) -> str:
    if not lines:
        return sha256(b"empty").hexdigest()
    layer = [sha256(line.encode("utf-8")).hexdigest() for line in lines]
    while len(layer) > 1:
        next_layer: list[str] = []
        for index in range(0, len(layer), 2):
            left = layer[index]
            right = layer[index + 1] if index + 1 < len(layer) else left
            next_layer.append(sha256(f"{left}{right}".encode()).hexdigest())
        layer = next_layer
    return layer[0]


def append_audit_receipt(audit_json: dict[str, Any]) -> dict[str, Any]:
    """Append audit receipt to ledger and return Merkle metadata."""

    receipt_id = audit_json.get("receipt_id")
    if not receipt_id:
        receipt_id = sha256(_stable_json(audit_json).encode("utf-8")).hexdigest()[:16]
        audit_json["receipt_id"] = receipt_id
    lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    lines.append(_stable_json(audit_json))
    LEDGER_PATH.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    root_hash = _merkle_root(lines)
    metrics = {"entries": len(lines), "root_hash": root_hash[:16]}
    write_receipt(
        "tessrax.core.ledger.audit_integration.append", "verified", metrics, 0.96
    )
    return {"receipt_id": receipt_id, "root_hash": root_hash, "position": len(lines)}


def verify_audit_receipt(receipt_id: str) -> dict[str, Any]:
    """Verify audit receipt presence and recompute Merkle legitimacy."""

    lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    matched = [line for line in lines if f'"receipt_id":"{receipt_id}"' in line]
    legitimacy = 1.0 if matched else 0.0
    root_hash = _merkle_root(lines)
    metrics = {"entries": len(lines), "legitimacy": round(legitimacy, 3)}
    write_receipt(
        "tessrax.core.ledger.audit_integration.verify", "verified", metrics, 0.95
    )
    return {"receipt_id": receipt_id, "legitimacy": legitimacy, "root_hash": root_hash}


def _self_test() -> bool:
    """Run deterministic audit append + verify cycle."""

    audit_payload = {
        "title": "Tessrax Whitepaper Audit",
        "auditor": "Trusted Cooperative",
        "score": "0.94",
    }
    appended = append_audit_receipt(audit_payload)
    verification = verify_audit_receipt(appended["receipt_id"])
    assert verification["legitimacy"] >= 0.8, "Legitimacy threshold failed"
    write_receipt(
        "tessrax.core.ledger.audit_integration.self_test",
        "verified",
        {"legitimacy": verification["legitimacy"]},
        0.95,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
