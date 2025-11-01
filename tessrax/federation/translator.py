"""
Protocol Translator — serializes and verifies contradiction packets for cross-instance exchange.
Emits 'FEDERATED_EXCHANGE' and 'FEDERATED_IMPORT' ledger events.
"""

from __future__ import annotations

import datetime
import hashlib
import json


def to_exchange_packet(record: dict, exchange_id: str, source: str, target: str) -> str:
    return json.dumps(
        {
            "exchange_id": exchange_id,
            "source_node": source,
            "target_node": target,
            "event_type": "FEDERATED_EXCHANGE",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "payload": record,
            "hash": hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest(),
        }
    )


def from_exchange_packet(packet: str) -> dict:
    data = json.loads(packet)
    payload = data.get("payload")
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if h != data.get("hash"):
        raise ValueError("hash mismatch")
    if isinstance(payload, dict):
        enriched = dict(payload)
        enriched.setdefault("event_type", "FEDERATED_IMPORT")
        return enriched
    return payload
