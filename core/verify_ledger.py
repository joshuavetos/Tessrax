"""Ledger verification helper with fallback search paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

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

    def compute_hash(prev_hash: str, payload: Mapping[str, Any]) -> str:
        serialised = json.dumps({"prev": prev_hash, "payload": payload}, sort_keys=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    previous_hash = "GENESIS"
    ok = True
    for index, entry in enumerate(_iter_entries(ledger_path), 1):
        entry_hash = entry.get("hash")
        if not isinstance(entry_hash, str):
            print(f"❌ Entry {index} missing hash field")
            ok = False
            break

        entry_prev = entry.get("prev_hash")
        if entry_prev is None:
            # Some historical ledgers omit explicit ``prev_hash`` on the
            # first entry; assume GENESIS for backwards compatibility.
            entry_prev = "GENESIS" if index == 1 else None
        if entry_prev is None:
            print(f"❌ Entry {index} missing prev_hash field")
            ok = False
            break
        if entry_prev != previous_hash:
            print(
                f"❌ Entry {index} prev_hash {entry_prev!r} does not match previous digest {previous_hash!r}"
            )
            ok = False
            break

        payload = entry.get("payload")
        if payload is None:
            payload = {k: v for k, v in entry.items() if k not in {"hash", "prev_hash"}}
        if not isinstance(payload, Mapping):
            print(f"❌ Entry {index} payload is not a mapping")
            ok = False
            break

        expected_hash = compute_hash(entry_prev, payload)
        if expected_hash != entry_hash:
            print(
                f"❌ Entry {index} hash mismatch (expected {expected_hash}, found {entry_hash})"
            )
            ok = False
            break

        previous_hash = entry_hash

    if ok:
        print(f"✅ Ledger chain verified successfully ({ledger_path})")
    return ok


if __name__ == "__main__":
    verify()
