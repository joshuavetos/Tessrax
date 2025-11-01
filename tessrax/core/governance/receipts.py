"""tessrax.core.governance.receipts
====================================

Governance utility helpers for Tessrax modules.

This module adheres to AEP-001, RVC-001, and POST-AUDIT-001 by
providing deterministic receipt handling utilities that can be
imported by higher-level modules without violating cold-start
assumptions. The utilities defined here avoid external dependencies
and remain compatible with Python 3.11.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

RECEIPT_DIR = (Path(__file__).resolve().parents[3] / "out" / "receipts").resolve()
RECEIPT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Receipt:
    """Immutable governance receipt with Tessrax identifiers.

    The receipt captures execution evidence in alignment with the
    Receipts-First Rule and double-lock verification directives. Each
    receipt contains an integrity hash computed from sorted metrics to
    support downstream Merkle stitching without introducing
    nondeterminism.
    """

    module: str
    status: str
    metrics: dict[str, Any]
    integrity_score: float
    runtime_info: str
    signature: str
    timestamp: str
    auditor: str = "Tessrax Governance Kernel v16"
    clauses: tuple[str, ...] = ("AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001")

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a plain dictionary."""

        return asdict(self)


def _stable_metrics_hash(metrics: dict[str, Any]) -> str:
    """Generate a deterministic SHA-256 hash for the provided metrics."""

    serialized = "|".join(f"{key}={metrics[key]}" for key in sorted(metrics))
    return sha256(serialized.encode("utf-8")).hexdigest()


def write_receipt(
    module: str, status: str, metrics: dict[str, Any], integrity: float
) -> Receipt:
    """Persist an execution receipt for the calling module.

    Parameters
    ----------
    module: str
        Module identifier in dotted form.
    status: str
        Textual status such as "verified" or "failed".
    metrics: Dict[str, Any]
        Deterministic metrics captured during runtime verification.
    integrity: float
        Integrity estimate between 0 and 1.

    Returns
    -------
    Receipt
        The structured receipt instance for additional auditing.
    """

    timestamp = datetime.now(timezone.utc).isoformat()
    runtime_info = "python-3.11"
    metrics_hash = _stable_metrics_hash(metrics)
    signature = sha256(
        f"{module}|{status}|{metrics_hash}|{timestamp}".encode()
    ).hexdigest()
    receipt = Receipt(
        module=module,
        status=status,
        metrics=metrics,
        integrity_score=integrity,
        runtime_info=runtime_info,
        signature=signature,
        timestamp=timestamp,
    )
    path = RECEIPT_DIR / f"{module.replace('.', '_')}_receipt.json"
    path.write_text(__receipt_to_json(receipt), encoding="utf-8")
    return receipt


def __receipt_to_json(receipt: Receipt) -> str:
    """Serialize a receipt using deterministic key ordering."""

    # Manual JSON encoding keeps dependencies minimal and deterministic.
    def _encode(value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(_encode(item) for item in value) + "]"
        if isinstance(value, dict):
            return (
                "{"
                + ",".join(f'"{k}":{_encode(value[k])}' for k in sorted(value))
                + "}"
            )
        raise TypeError(f"Unsupported value type: {type(value)!r}")

    content = receipt.to_dict()
    json_body = (
        "{"
        + ",".join(f'"{key}":{_encode(content[key])}' for key in sorted(content))
        + "}"
    )
    return json_body + "\n"


__all__ = ["Receipt", "write_receipt"]
