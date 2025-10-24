"""CLI wrapper for Tessrax ledger verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from . import verify_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Tessrax ledger receipts")
    parser.add_argument("path", type=Path, help="Path to ledger JSONL file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        verify_file(args.path)
    except Exception as exc:  # pragma: no cover - surfaced directly to CLI
        raise SystemExit(f"Ledger verification failed: {exc}")

    print("Ledger verified: integrity intact.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
