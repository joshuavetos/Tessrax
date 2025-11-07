#!/usr/bin/env python3
"""Ledger utility CLI for appending file digests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ledger import append as ledger_append


def append_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Cannot append missing file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {
        "timestamp": time.time(),
        "file": str(path),
        "sha256": digest,
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
    }
    ledger_append(entry)
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append records to the Tessrax ledger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    append_parser = subparsers.add_parser("append", help="Append a file digest to the ledger")
    append_parser.add_argument("--file", type=Path, required=True, help="File to hash and append")

    args = parser.parse_args(argv)

    if args.command == "append":
        receipt = append_file(args.file)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    raise SystemExit("Unknown command")


if __name__ == "__main__":  # pragma: no cover - CLI behavior
    raise SystemExit(main())
