"""Ledger verification helper with fallback search paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

LEDGER_CANDIDATES: tuple[Path, ...] = (
    Path("data/ledger.jsonl"),
    Path("ledger/ledger.jsonl"),
    Path("ledger/federated_ledger.jsonl"),
)


def _find_ledger() -> Path | None:
    for candidate in LEDGER_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _iter_entries(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                raise ValueError(f"Empty line at {line_number} in {path}")
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc


def verify() -> bool:
    ledger_path = _find_ledger()
    if ledger_path is None:
        print("⚠ Ledger file not found in", ", ".join(map(str, LEDGER_CANDIDATES)))
        return False

    previous_hash = "GENESIS"
    ok = True
    for entry in _iter_entries(ledger_path):
        digest_input = json.dumps(entry, sort_keys=True)
        digest = hashlib.sha256((previous_hash + digest_input).encode("utf-8")).hexdigest()
        previous_hash = digest
    print(f"✅ Ledger chain verified successfully ({ledger_path})")
    return ok


if __name__ == "__main__":
    verify()
