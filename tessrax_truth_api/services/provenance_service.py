"""Merkle style provenance tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, status

from tessrax_truth_api.utils import (
    ensure_directory,
    hmac_signature,
    load_env,
    merkle_hash,
    receipt_identifier,
    utcnow,
    verify_signature,
)


@dataclass
class ReceiptRecord:
    uuid: str
    timestamp: str
    payload: dict[str, any]
    signature: str
    prev_hash: str | None
    merkle_hash: str


class ProvenanceService:
    """Append only ledger manager for Truth API receipts."""

    def __init__(self, ledger_path: Path | None = None) -> None:
        env = load_env()
        configured = env.get("TRUTH_API_LEDGER_PATH")
        self._ledger_path = (
            Path(configured)
            if configured
            else (ledger_path or Path("ledger/truth_api_ledger.jsonl"))
        )
        ensure_directory(self._ledger_path)

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    def _load_entries(self) -> list[dict[str, any]]:
        if not self._ledger_path.exists():
            return []
        with self._ledger_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _persist_entries(self, entries: list[dict[str, any]]) -> None:
        with self._ledger_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def append_receipt(
        self, payload: dict[str, any], *, seed: str | None = None
    ) -> ReceiptRecord:
        entries = self._load_entries()
        prev_hash = entries[-1]["merkle_hash"] if entries else None
        receipt_uuid = receipt_identifier(seed=seed)

        for entry in entries:
            if entry["uuid"] == receipt_uuid:
                return ReceiptRecord(**entry)

        timestamp = utcnow().isoformat()
        signature = hmac_signature(payload)
        merkle = merkle_hash(payload=payload, prev_hash=prev_hash)
        record = {
            "uuid": receipt_uuid,
            "timestamp": timestamp,
            "payload": payload,
            "signature": signature,
            "prev_hash": prev_hash,
            "merkle_hash": merkle,
        }
        entries.append(record)
        with self._ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return ReceiptRecord(**record)

    def verify_receipt(self, uuid: str) -> ReceiptRecord:
        entries = self._load_entries()
        for idx, entry in enumerate(entries):
            if entry["uuid"] == uuid:
                prev_hash = entries[idx - 1]["merkle_hash"] if idx > 0 else None
                expected_hash = merkle_hash(
                    payload=entry["payload"], prev_hash=prev_hash
                )
                if entry["merkle_hash"] != expected_hash:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Merkle hash mismatch",
                    )
                if not verify_signature(entry["payload"], entry["signature"]):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Signature mismatch",
                    )
                return ReceiptRecord(**entry)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Receipt not found"
        )

    def record_event(self, event_type: str, details: dict[str, any]) -> ReceiptRecord:
        payload = {"event_type": event_type, "details": details}
        return self.append_receipt(payload)
