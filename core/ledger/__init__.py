"""Ledger façade providing convenient append/verify helpers for runtime tests."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from tessrax.ledger import (
    GENESIS_HASH,
    Ledger,
    LedgerReceipt,
    build_cli,
    compute_merkle_root,
    verify_file,
)

from ..governance_kernel import last_decision

__all__ = [
    "GENESIS_HASH",
    "Ledger",
    "LedgerReceipt",
    "build_cli",
    "compute_merkle_root",
    "verify_file",
    "append",
    "verify",
]


_DEFAULT_LEDGER_PATH = Path("data/ledger.jsonl")
_DEFAULT_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
_DEFAULT_LEDGER = Ledger(_DEFAULT_LEDGER_PATH)


def append(decision_payload: Mapping[str, object]) -> dict[str, object]:
    """Append the most recent governance decision to the ledger.

    Parameters
    ----------
    decision_payload:
        The DLK-verified decision summary previously returned by
        :func:`core.governance_kernel.resolve`.  The mapping must contain a
        ``decision_id`` field that matches the stored decision envelope.

    Returns
    -------
    dict
        Serialised ledger receipt suitable for JSON logging.
    """

    decision_id = decision_payload.get("decision_id") if isinstance(decision_payload, Mapping) else None
    if not decision_id:
        raise ValueError("Decision payload must include a decision_id for ledger persistence")

    decision = last_decision()
    if decision is None or str(decision.decision_id) != str(decision_id):
        raise ValueError("No matching governance decision cached for ledger append")

    receipt = _DEFAULT_LEDGER.append(decision)
    payload = receipt.to_json()
    payload.update(
        {
            "auditor": "Tessrax Governance Kernel v16",
            "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
            "status": "DLK-VERIFIED",
            "timestamp": decision_payload.get("timestamp"),
            "runtime_info": {
                "ledger_path": str(_DEFAULT_LEDGER_PATH),
                "prev_hash": payload.get("prev_hash"),
                "hash": payload.get("hash"),
            },
        }
    )
    return payload


def verify() -> bool:
    """Verify the integrity of the in-memory ledger chain."""

    _DEFAULT_LEDGER.verify()
    return True
