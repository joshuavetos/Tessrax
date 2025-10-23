"""Lightweight audit ledger for Tessrax demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

from .types import GovernanceDecision, LedgerReceipt

GENESIS_HASH = "GENESIS"


class Ledger:
    """Append-only ledger with hash chaining."""

    def __init__(self) -> None:
        self._receipts: List[LedgerReceipt] = []

    def append(self, decision: GovernanceDecision, signature: Optional[str] = None) -> LedgerReceipt:
        prev_hash = self._receipts[-1].hash if self._receipts else GENESIS_HASH
        payload = decision.to_summary()
        digest = self._hash_payload(prev_hash, payload)
        receipt = LedgerReceipt(
            decision=decision,
            prev_hash=prev_hash,
            hash=digest,
            signature=signature,
        )
        self._receipts.append(receipt)
        return receipt

    def receipts(self) -> List[LedgerReceipt]:
        return list(self._receipts)

    def verify(self, receipts: Optional[Iterable[LedgerReceipt]] = None) -> bool:
        chain = list(receipts if receipts is not None else self._receipts)
        prev_hash = GENESIS_HASH
        for receipt in chain:
            expected = self._hash_payload(prev_hash, receipt.decision.to_summary())
            if expected != receipt.hash:
                raise ValueError(f"Ledger hash mismatch for decision {receipt.decision.contradiction.claim_a.claim_id}")
            prev_hash = receipt.hash
        return True

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for receipt in self._receipts:
                handle.write(json.dumps(receipt.to_json(), sort_keys=True) + "\n")

    @staticmethod
    def _hash_payload(prev_hash: str, payload: dict) -> str:
        serialised = json.dumps({"prev": prev_hash, "payload": payload}, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialised).hexdigest()


def verify_file(path: Path) -> bool:
    """Verify ledger receipts stored in a JSONL file."""

    prev_hash = GENESIS_HASH
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            summary = {
                key: payload[key]
                for key in (
                    "event_type",
                    "timestamp",
                    "action",
                    "severity",
                    "clarity_fuel",
                    "subject",
                    "metric",
                    "rationale",
                )
            }
            expected = Ledger._hash_payload(prev_hash, summary)
            if expected != payload["hash"]:
                raise ValueError("Ledger hash mismatch on disk receipt")
            prev_hash = payload["hash"]
    return True


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Verify Tessrax ledger receipts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="Verify a JSONL ledger file")
    verify_parser.add_argument("path", type=Path, help="Path to ledger JSONL file")
    args = parser.parse_args()

    if args.command == "verify":
        try:
            verify_file(args.path)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            raise SystemExit(f"Ledger verification failed: {exc}")
        print("Ledger verified: integrity intact.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
