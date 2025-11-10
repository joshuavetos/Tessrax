"""Merkle-backed ledger append implementation."""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Any

_LOG_PATH = pathlib.Path("ledger/logs/merkle.log")


def _serialize(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    try:
        return json.dumps(entry, default=str)
    except TypeError:
        return repr(entry)


def append_merkle(entry: Any) -> dict[str, Any]:
    """Append the entry to the merkle log with a digest receipt."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize(entry)
    serialized_bytes = serialized.encode("utf-8")
    digest = hashlib.sha256(serialized_bytes).hexdigest()
    timestamp = time.time()

    persisted_record = {
        "mode": "merkle",
        "entry": serialized,
        "hash": digest,
        "timestamp": timestamp,
    }

    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(persisted_record) + "\n")

    return {
        "mode": "merkle",
        "entry": entry,
        "hash": digest,
        "timestamp": timestamp,
        "serialized": serialized,
    }
