"""Audit capsule validating ledger Merkle roots."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List

AUDIT_METADATA = {
    "auditor": "Tessrax Governance Kernel v16",
    "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
}


class AuditCapsule:
    """Verifies the integrity of receipt collections via Merkle hashing."""

    def verify(self, receipts: Iterable[Dict[str, object]], expected_root: str | None) -> Dict[str, object]:
        """Compute a Merkle root and compare it to ``expected_root``."""

        receipt_ids: List[str] = []
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise TypeError("Receipts must be dictionaries")
            receipt_id = str(receipt.get("receipt_id"))
            if len(receipt_id) != 64:
                raise ValueError("Receipt IDs must be 64-character SHA-256 hex strings")
            receipt_ids.append(receipt_id)

        computed_root = self._merkle(receipt_ids)
        status = computed_root == expected_root
        return {
            "status": status,
            "computed_root": computed_root,
            "expected_root": expected_root,
            "audit": {**AUDIT_METADATA},
        }

    def _merkle(self, receipt_ids: List[str]) -> str | None:
        if not receipt_ids:
            return None
        current_level = receipt_ids
        while len(current_level) > 1:
            next_level: List[str] = []
            for index in range(0, len(current_level), 2):
                pair = current_level[index:index + 2]
                if len(pair) == 1:
                    pair = (pair[0], pair[0])
                concatenated = "".join(pair)
                next_level.append(hashlib.sha256(concatenated.encode("utf-8")).hexdigest())
            current_level = next_level
        return current_level[0]
