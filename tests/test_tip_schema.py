from __future__ import annotations

from datetime import datetime, timezone

import jsonschema

from tessrax.tip import TIPHandshake, TIP_PROTOCOL, load_tip_schema


def test_tip_handshake_to_payload_matches_schema() -> None:
    handshake = TIPHandshake(
        source="core.orchestrator",
        target="ai.skills",
        nonce="abcdef123456",
        capabilities=["receipts", "metrics"],
        signature="f" * 64,
        integrity=0.95,
        status="accepted",
    )
    payload = handshake.to_payload()
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    schema = load_tip_schema()
    jsonschema.validate(instance=payload, schema=schema)
    assert payload["protocol"] == TIP_PROTOCOL
