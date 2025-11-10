"""CLI entry point for governed Merkle proof verification (DLK-verified)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from tessrax.core import merkle_proof
from tessrax.core.merkle_engine import hash_receipt as engine_hash_receipt
from tessrax.ledger import verify_file

_DEMO_RECEIPT = (
    Path(__file__).resolve().parents[1]
    / "ledger"
    / "receipts"
    / "ethical_drift_v17_5.jsonl"
)


def _load_receipt(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Receipt file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Receipt payload must be a JSON object")
    return payload


def _verify_json_receipt(path: Path) -> None:
    payload = _load_receipt(path)
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


def _verify_demo_ledger() -> None:
    if not _DEMO_RECEIPT.exists():
        raise FileNotFoundError(
            f"Demo ledger fixture missing; expected {_DEMO_RECEIPT}"
        )
    verify_file(_DEMO_RECEIPT)
    print(
        "Demo ledger verification succeeded for ethical_drift_v17_5.jsonl",
        file=sys.stdout,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessrax.verify",
        description="Verify Tessrax ledger receipts using governed Merkle proofs.",
    )
    parser.add_argument(
        "receipt_file",
        nargs="?",
        help="Path to the governed ledger receipt JSON file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run verification against the bundled ethical_drift_v17_5.jsonl ledger",
    )
    return parser


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.demo and not args.receipt_file:
        parser.print_help(sys.stderr)
        return 2

    try:
        if args.demo:
            _verify_demo_ledger()
        else:
            target = Path(args.receipt_file).expanduser().resolve()
            if target.suffix.lower() == ".jsonl":
                verify_file(target)
                print(
                    f"Ledger verification succeeded for {target}",
                    file=sys.stdout,
                )
            else:
                _verify_json_receipt(target)
        return 0
    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _cli(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
