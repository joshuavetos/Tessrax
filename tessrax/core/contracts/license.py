"""License embedding utilities for Tessrax ledgers."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

LICENSE_LEDGER = Path("ledger/license_ledger.jsonl")


def embed_license(event: dict, license_id: str = "Tessrax-Covenant-1.0"):
    """Attach SPDX-style license metadata to a ledger entry."""
    wrapped = {
        "event": "licensed_entry",
        "license": license_id,
        "payload": event,
        "timestamp": time.time(),
        "hash": hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest(),
    }
    LICENSE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LICENSE_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(wrapped) + "\n")
    return wrapped
