"""Ledger helpers exposed for Tessrax core modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ledger_persistence import append_entry as _root_append_entry

DEFAULT_LEDGER_PATH = Path("ledger.jsonl")


def append_entry(entry: dict[str, Any], path: Path | str | None = None) -> None:
    """Append an entry to the ledger, ensuring directories exist."""

    if path is None:
        _root_append_entry(entry)
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle, ensure_ascii=False)
        handle.write("\n")


__all__ = ["append_entry", "DEFAULT_LEDGER_PATH"]
