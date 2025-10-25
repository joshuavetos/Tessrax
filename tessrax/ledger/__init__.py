"""Lightweight audit ledger for Tessrax demos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from tessrax.types import GovernanceDecision, LedgerReceipt

GENESIS_HASH = "GENESIS"

__all__ = [
    "GENESIS_HASH",
    "Ledger",
    "LedgerReceipt",
    "verify_file",
    "build_cli",
    "compute_merkle_root",
]


def compute_merkle_root(batch: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Compute recursive Merkle root for a batch of ledger entries."""

    if not batch:
        return None
    nodes = [
        hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()
        for entry in batch
    ]
    while len(nodes) > 1:
        paired: List[str] = []
        for index in range(0, len(nodes), 2):
            left = nodes[index]
            right = nodes[index + 1] if index + 1 < len(nodes) else nodes[index]
            paired.append(hashlib.sha256((left + right).encode("utf-8")).hexdigest())
        nodes = paired
    return nodes[0]


class Ledger:
    """Append-only ledger with hash chaining."""

    def __init__(self) -> None:
        self._receipts: List[LedgerReceipt] = []
        self._meta_events: List[dict[str, Any]] = []

    def append(
        self,
        decision: GovernanceDecision,
        signature: Optional[str] = None,
        *,
        sub_merkle_root: Optional[str] = None,
    ) -> LedgerReceipt:
        prev_hash = self._receipts[-1].hash if self._receipts else GENESIS_HASH
        payload = decision.to_summary()
        digest = self._hash_payload(prev_hash, payload)
        receipt = LedgerReceipt(
            decision=decision,
            prev_hash=prev_hash,
            hash=digest,
            signature=signature,
            sub_merkle_root=sub_merkle_root,
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
                raise ValueError(
                    "Ledger hash mismatch for decision "
                    f"{receipt.decision.contradiction.claim_a.claim_id}"
                )
            prev_hash = receipt.hash
        return True

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for receipt in self._receipts:
                handle.write(json.dumps(receipt.to_json(), sort_keys=True) + "\n")

    def append_meta(self, channel: str, payload: Mapping[str, Any]) -> None:
        """Record supplementary ledger metadata events."""

        event = {"channel": channel, **dict(payload)}
        if event.get("source") == "adversarial_generator":
            meta = event.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["synthetic"] = True
            event["meta"] = meta
        event.setdefault("event_type", event.get("event_type", "UNKNOWN"))
        self._meta_events.append(event)

    def meta_events(self) -> List[dict[str, Any]]:
        """Expose recorded metadata events for downstream telemetry."""

        return list(self._meta_events)

    def append_batch(
        self,
        decisions: Iterable[GovernanceDecision],
        *,
        consensus_nodes: Optional[Iterable[Any]] = None,
    ) -> Optional[str]:
        """Append a batch of decisions and emit a sub-Merkle root for verification."""

        decisions_list = list(decisions)
        if not decisions_list:
            return None
        payloads = [decision.to_summary() for decision in decisions_list]
        root = compute_merkle_root(payloads)
        for decision in decisions_list:
            self.append(decision, sub_merkle_root=root)
        if consensus_nodes:
            for node in consensus_nodes:
                if hasattr(node, "commit"):
                    node.commit(root)
        return root

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
                    "decision_id",
                    "action",
                    "severity",
                    "clarity_fuel",
                    "subject",
                    "metric",
                    "rationale",
                    "confidence",
                    "contradiction_ids",
                    "synthesis",
                    "protocol",
                    "timestamp_token",
                    "signature",
                )
                if key in payload
            }
            expected = Ledger._hash_payload(prev_hash, summary)
            if expected != payload["hash"]:
                raise ValueError("Ledger hash mismatch on disk receipt")
            prev_hash = payload["hash"]
    return True


def build_cli(argv: Optional[Sequence[str]] = None) -> argparse.ArgumentParser:
    """Create the canonical ledger CLI parser."""

    parser = argparse.ArgumentParser(description="Verify Tessrax ledger receipts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify", help="Verify a JSONL ledger file")
    verify_parser.add_argument("path", type=Path, help="Path to ledger JSONL file")
    return parser


def _cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_cli(argv)
    args = parser.parse_args(argv)

    if args.command == "verify":
        from tessrax.ledger.verify import main as verify_main

        verify_main([str(args.path)])


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry-point used by ``python -m tessrax.ledger``."""

    _cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()
