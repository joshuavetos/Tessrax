"""Merkle-backed ledger append implementation."""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any, Dict

_LOG_PATH = pathlib.Path("ledger/logs/merkle.log")


def _serialize(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    try:
        return json.dumps(entry, default=str)
    except TypeError:
        return repr(entry)


def append_merkle(entry: Any) -> Dict[str, Any]:
    """Append the entry to the merkle log with a digest receipt."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize(entry)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    record = {
        "mode": "merkle",
        "entry": entry,
        "hash": digest,
        "timestamp": time.time(),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return record
