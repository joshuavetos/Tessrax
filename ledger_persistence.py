"""Append-only JSONL ledger persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

LEDGER_PATH = "ledger.jsonl"


def _ensure_ledger_exists() -> Path:
    path = Path(LEDGER_PATH)
    if not path.exists():
        path.touch()
    return path


def append_entry(entry: dict) -> None:
    """Append a JSON serialised entry to the ledger file."""

    path = _ensure_ledger_exists()
    with path.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.write("\n")


def load_ledger(limit: int = 50) -> list[dict]:
    """Load up to ``limit`` most recent entries from the ledger."""

    path = _ensure_ledger_exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    # Return the newest entries first.
    selected = lines[-limit:][::-1]
    return [json.loads(line) for line in selected]
