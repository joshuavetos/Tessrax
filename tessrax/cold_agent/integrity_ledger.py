"""Integrity ledger with Merkle root computation for Cold Agent receipts."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}


class IntegrityLedger:
    """Append-only ledger storing Cold Agent receipts."""

    def __init__(self) -> None:
        self.receipts: List[Dict[str, object]] = []

    def append(self, receipt: Dict[str, object]) -> None:
        """Append a receipt to the ledger with runtime verification."""

        if not isinstance(receipt, dict):
            raise TypeError("receipt must be a dictionary")
        required_fields = {"receipt_id", "signature", "audit"}
        missing = required_fields.difference(receipt)
        if missing:
            raise KeyError(f"Receipt missing required fields: {sorted(missing)}")
        self.receipts.append(receipt)

    def merkle_root(self) -> str | None:
        """Compute the Merkle root of the stored receipt identifiers."""

        hashes = [str(receipt["receipt_id"]) for receipt in self.receipts]
        if not hashes:
            return None
        current_level: Sequence[str] = hashes
        while len(current_level) > 1:
            next_level: List[str] = []
            for index in range(0, len(current_level), 2):
                pair = current_level[index:index + 2]
                if len(pair) == 1:
                    pair = (pair[0], pair[0])
                concatenated = "".join(pair)
                digest = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()
                next_level.append(digest)
            current_level = next_level
        return current_level[0]

    def replay(self) -> List[Dict[str, object]]:
        """Return a shallow copy of stored receipts for auditing."""

        return [dict(receipt) for receipt in self.receipts]

    def audit_metadata(self) -> Dict[str, object]:
        """Return governance metadata for external receipts."""

        return {**AUDIT_METADATA, "entries": len(self.receipts)}
