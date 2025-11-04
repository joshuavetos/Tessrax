"""CLI wrapper for Tessrax ledger verification."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tessrax.ledger import verify_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Tessrax ledger receipts")
    parser.add_argument("path", type=Path, nargs='?', default=Path('ledger/receipts/ethical_drift_v17_5.jsonl'), help="Path to ledger JSONL file")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    target_path = args.path
    if not target_path.exists():
        raise SystemExit(f'Ledger verification failed: missing file {target_path}')
    try:
        verify_file(target_path)
    except Exception as exc:  # pragma: no cover - surfaced directly to CLI
        raise SystemExit(f"Ledger verification failed: {exc}")

    print("Ledger verified: integrity intact.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
