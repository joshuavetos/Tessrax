"""Lightweight reconciliation engine for governance compliance harnesses.

The implementation intentionally mirrors a subset of the behaviour found
in :mod:`tessrax.metabolism.reconcile` but avoids pulling in the heavy
governance dependencies.  The engine produces deterministic receipts
with embedded governance metadata so that Merkle equivalence checks can
be performed during the federated harness tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ClarityEnvelope:
    """Container describing a single clarity receipt produced by the engine."""

    payload: Mapping[str, object]

    def to_receipt(self) -> dict[str, object]:
        """Return a dictionary copy of the payload for persistence."""

        return dict(self.payload)


class ReconciliationEngine:
    """Generate clarity envelopes from simple contradiction records.

    Parameters
    ----------
    ledger:
        Ledger-like object that exposes an ``append`` method accepting a
        mapping.  The harness injects the lightweight ledger from
        :mod:`tessrax.core.ledger` but the contract is deliberately
        duck-typed to keep the engine standalone.
    """

    def __init__(self, ledger: object) -> None:
        if not hasattr(ledger, "append"):
            raise TypeError("ledger must provide an append() method")
        self._ledger = ledger

    # ------------------------------------------------------------------
    # Core reconciliation logic
    # ------------------------------------------------------------------
    def reconcile(self, contradictions: Sequence[Mapping[str, object]]) -> ClarityEnvelope:
        """Transform a batch of contradictions into a clarity envelope.

        The function enforces runtime validation (``RVC-001``) by
        ensuring each contradiction contains ``subject``, ``metric`` and
        ``value`` fields.  The resulting payload contains a deterministic
        hash that is used for Merkle comparisons in the federation
        harness.
        """

        if not contradictions:
            raise ValueError("at least one contradiction is required")

        validated: list[dict[str, object]] = []
        for index, contradiction in enumerate(contradictions):
            if not isinstance(contradiction, Mapping):
                raise TypeError("contradictions must be mappings")
            missing = [
                key
                for key in ("subject", "metric", "value")
                if key not in contradiction
            ]
            if missing:
                raise ValueError(f"contradiction {index} missing fields: {missing}")
            validated.append(dict(contradiction))

        payload = {
            "directive": "CLARITY_RESOLUTION",
            "timestamp": _timestamp(),
            "auditor": "Tessrax Governance Kernel v16",
            "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
            "contradictions": validated,
        }
        payload["hash"] = self._compute_receipt_hash(payload)
        self._ledger.append(payload)
        return ClarityEnvelope(payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_receipt_hash(payload: Mapping[str, object]) -> str:
        serialised = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


__all__ = ["ClarityEnvelope", "ReconciliationEngine"]
