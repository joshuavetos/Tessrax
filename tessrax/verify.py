"""CLI entry point for governed Merkle proof verification (DLK-verified)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tessrax.core.merkle_engine import hash_receipt as engine_hash_receipt
from tessrax.core import merkle_proof


def _load_receipt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Receipt file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Receipt payload must be a JSON object")
    return payload


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tessrax.verify",
        description="Verify Tessrax ledger receipts using governed Merkle proofs.",
    )
    parser.add_argument("receipt_file", help="Path to the governed ledger receipt JSON file")
    args = parser.parse_args(argv)
    receipt_path = Path(args.receipt_file).expanduser().resolve()
    payload = _load_receipt(receipt_path)
    receipt_id = payload.get("receipt_id")
    if receipt_id is None:
        receipt_id = engine_hash_receipt(payload)
    bundle = merkle_proof.generate_proof(str(receipt_id))
    payload_hash = engine_hash_receipt(payload)
    if payload_hash != bundle.leaf_hash:
        raise ValueError("Receipt payload hash mismatch; ledger integrity compromised")
    merkle_proof.verify_proof(bundle, bundle.merkle_root)
    print(
        f"Verification success for receipt {bundle.receipt_id} (root: {bundle.merkle_root})",
        file=sys.stdout,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _cli(argv)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
