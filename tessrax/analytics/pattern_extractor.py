"""
Ledger Pattern Extractor — mines historical ledger data to discover probabilistic resolution patterns.
Emits 'PREDICTIVE_PATTERN' records when invoked with --emit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path


def extract_patterns_from_stream(lines: Iterable[str]) -> dict[str, dict[str, float]]:
    counter: dict[str, Counter[str]] = defaultdict(Counter)
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ctype = rec.get("contradiction_type") or rec.get("event_type")
        resolution = rec.get("resolution")
        outcome = (
            resolution.get("result") if isinstance(resolution, dict) else resolution
        )
        if ctype and outcome:
            counter[ctype][outcome] += 1
    patterns: dict[str, dict[str, float]] = {}
    for ctype, ctr in counter.items():
        total = sum(ctr.values())
        patterns[ctype] = {o: round(n / total, 3) for o, n in ctr.items() if total}
    return patterns


def _load_lines(path: Path | None) -> list[str]:
    if path is None:
        if sys.stdin.isatty():
            return []
        return [line for line in sys.stdin.read().splitlines() if line]
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ledger resolution patterns")
    parser.add_argument(
        "--source", type=Path, help="Path to ledger JSONL source", default=None
    )
    parser.add_argument(
        "--emit", action="store_true", help="Emit predictive pattern record"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(list(argv) if argv is not None else None)
    lines = _load_lines(args.source)
    patterns = extract_patterns_from_stream(lines)
    if args.emit:
        event = {
            "event_type": "PREDICTIVE_PATTERN",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "patterns": patterns,
            "source": str(args.source) if args.source else "stdin",
        }
        json.dump(event, sys.stdout)
        sys.stdout.write("\n")
    else:
        json.dump(patterns, sys.stdout, indent=2)
        sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
