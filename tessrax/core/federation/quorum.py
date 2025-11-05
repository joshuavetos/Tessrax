"""Quorum aggregation utilities for the Tessrax federation layer."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable


def merge_events(events: Iterable[dict]) -> dict:
    """Merge contradiction events into a federated quorum record.

    The function enforces basic structural validation to satisfy the
    Runtime Verification Clause (RVC-001). If any event is missing the
    ``hash`` or ``node`` keys, a :class:`KeyError` is raised, preventing
    ambiguous ledger writes.
    """

    events_list = list(events)
    if not events_list:
        raise ValueError("No events supplied for quorum aggregation")

    for event in events_list:
        if "hash" not in event or "node" not in event:
            raise KeyError("Event payload lacks required hash/node fields")

    sorted_events = sorted(events_list, key=lambda e: (e["hash"], e["node"]))
    joined = json.dumps(sorted_events, sort_keys=True).encode()
    qhash = hashlib.sha256(joined).hexdigest()
    return {
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
        "event": "federated_quorum",
        "members": [e["node"] for e in sorted_events],
        "count": len(sorted_events),
        "qhash": qhash,
    }


__all__ = ["merge_events"]
