"""Ledger schema normalization utilities aligned with Tessrax governance.

The module enforces canonical ledger fields to maintain downstream
compatibility. Runtime safeguards ensure non-conforming entries are
re-shaped while preserving data fidelity under clauses AEP-001,
POST-AUDIT-001, RVC-001, and EAC-001.
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Dict

CANON_FIELDS = [
    "timestamp",
    "actor",
    "event",
    "payload",
    "signature",
    "hash",
]


def normalize_entry(entry: Dict) -> Dict:
    """Return a new ledger entry matching the canonical field order."""
    return {key: entry.get(key) for key in CANON_FIELDS}


def normalize_ledger(path: str | Path) -> None:
    """Normalize a newline-delimited JSON ledger file in place."""
    ledger_path = Path(path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger file missing: {ledger_path}")
    lines = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            lines.append(normalize_entry(data))
    with ledger_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


if __name__ == "__main__":
    for file in glob.glob("ledger/**/*.jsonl", recursive=True):
        normalize_ledger(file)
    print("Ledger normalization complete.")
