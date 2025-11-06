"""Ledger helpers exposed for Tessrax core modules.

This module provides two layers of ledger functionality:

``append_entry``
    Lightweight bridge into :mod:`ledger_persistence` used across legacy
    integration tests.

``Ledger``
    Governance-aware append-only JSONL ledger designed for the
    governance compliance harnesses introduced in the Tessrax
    Federated Test Harness v18 series.  The class intentionally avoids
    any dependency on the heavier :mod:`tessrax.ledger` governance
    objects so that cold Python environments can exercise the
    federation, key vault, and integrity monitoring flows without
    importing the full governance kernel.

Both utilities include runtime verification hooks in line with the
Tessrax governance clauses (``AEP-001``, ``RVC-001``, ``POST-AUDIT-001``,
and ``DLK-001``).  The ledger records every write with a chain hash to
ensure that tampering can be detected during Merkle reconciliation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ledger_persistence import append_entry as _root_append_entry

DEFAULT_LEDGER_PATH = Path("ledger.jsonl")


def append_entry(entry: dict[str, Any], path: Path | str | None = None) -> None:
    """Append an entry to the ledger, ensuring directories exist."""

    if path is None:
        _root_append_entry(entry)
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle, ensure_ascii=False)
        handle.write("\n")


@dataclass(frozen=True)
class LedgerReceipt:
    """Immutable projection of a ledger entry.

    Parameters
    ----------
    payload:
        Original JSON-serialisable mapping supplied to
        :meth:`Ledger.append`.
    hash:
        Chain hash derived from ``payload`` and the ``prev_hash`` field.
    prev_hash:
        Chain hash of the previous receipt, or ``"GENESIS"`` for the
        first entry.
    timestamp:
        UTC timestamp indicating when the receipt was persisted.
    """

    payload: Mapping[str, Any]
    hash: str
    prev_hash: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the receipt."""

        return {
            "payload": dict(self.payload),
            "hash": self.hash,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
        }


class Ledger:
    """Append-only JSONL ledger with chain hashes and verification hooks.

    The implementation honours the Tessrax governance directives by
    storing an explicit provenance trail (``auditor`` and ``clauses``)
    for every entry.  Consumers may access a defensive copy of the raw
    receipts via :meth:`receipts` for Merkle equivalence checks.
    """

    _GENESIS_HASH = "GENESIS"

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._receipts: list[LedgerReceipt] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                self._load_from_disk()
            else:
                self._path.touch()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_from_disk(self) -> None:
        if self._path is None:
            return
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                receipt = LedgerReceipt(
                    payload=data["payload"],
                    hash=data["hash"],
                    prev_hash=data.get("prev_hash", self._GENESIS_HASH),
                    timestamp=data.get("timestamp", ""),
                )
                self._receipts.append(receipt)

    @staticmethod
    def _compute_hash(prev_hash: str, payload: Mapping[str, Any]) -> str:
        serialised = json.dumps({"prev": prev_hash, "payload": payload}, sort_keys=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def append(self, entry: Mapping[str, Any]) -> LedgerReceipt:
        """Append a mapping to the ledger and return the resulting receipt.

        Runtime verification (``RVC-001``) ensures the input is JSON
        serialisable and contains the governance identity required by the
        Tessrax directives.  Missing governance metadata is injected so
        downstream audits remain uniform.
        """

        if not isinstance(entry, Mapping):
            raise TypeError("Ledger entries must be mappings")

        payload = dict(entry)
        payload.setdefault("auditor", "Tessrax Governance Kernel v16")
        payload.setdefault(
            "clauses",
            ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
        )
        prev_hash = self._receipts[-1].hash if self._receipts else self._GENESIS_HASH
        chain_hash = self._compute_hash(prev_hash, payload)
        timestamp = payload.setdefault("timestamp", self._timestamp())
        receipt = LedgerReceipt(payload=payload, hash=chain_hash, prev_hash=prev_hash, timestamp=timestamp)
        self._receipts.append(receipt)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt.to_dict(), ensure_ascii=False) + "\n")
        return receipt

    def append_many(self, entries: Iterable[Mapping[str, Any]]) -> list[LedgerReceipt]:
        """Append a collection of entries, returning their receipts."""

        return [self.append(entry) for entry in entries]

    def receipts(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded receipts as plain dictionaries."""

        return [receipt.to_dict() for receipt in self._receipts]

    def verify(self) -> bool:
        """Validate the hash chain for all receipts.

        Any detected mismatch raises ``ValueError`` to satisfy the
        ``RVC-001`` directive.  Successful verification returns ``True``.
        """

        prev_hash = self._GENESIS_HASH
        for receipt in self._receipts:
            expected = self._compute_hash(prev_hash, receipt.payload)
            if expected != receipt.hash:
                raise ValueError("Ledger hash mismatch detected")
            prev_hash = receipt.hash
        return True


__all__ = ["append_entry", "DEFAULT_LEDGER_PATH", "Ledger", "LedgerReceipt"]
