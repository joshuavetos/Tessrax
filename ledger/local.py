"""Local ledger append implementation."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

_LOG_PATH = pathlib.Path("ledger/logs/local.log")


def _serialize(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    try:
        return json.dumps(entry, default=str)
    except TypeError:
        return repr(entry)


def append_local(entry: Any) -> dict[str, Any]:
    """Append the entry to the local log and return a receipt."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": "local",
        "entry": entry,
        "timestamp": time.time(),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(_serialize(record) + "\n")
    return record
