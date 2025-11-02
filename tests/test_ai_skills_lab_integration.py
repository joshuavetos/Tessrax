from __future__ import annotations

import json
from pathlib import Path
import pytest

from ai_skills.lab_integration import AISkillsLabClient, LiveReceiptStream, ReceiptEnvelope


@pytest.fixture()
def sample_ledger(tmp_path: Path) -> Path:
    ledger = tmp_path / "ledger.jsonl"
    records = [
        {
            "uuid": "receipt-1",
            "timestamp": "2025-01-01T00:00:00Z",
            "payload": {"claim": "alpha"},
            "signature": "s" * 64,
            "prev_hash": None,
            "merkle_hash": "m" * 64,
        },
        {
            "uuid": "receipt-2",
            "timestamp": "2025-01-01T00:01:00Z",
            "payload": {"claim": "beta"},
            "signature": "t" * 64,
            "prev_hash": "m" * 64,
            "merkle_hash": "n" * 64,
        },
    ]
    ledger.write_text("\n".join(json.dumps(entry) for entry in records) + "\n", encoding="utf-8")
    return ledger


def test_live_receipt_stream_reads_existing_entries(sample_ledger: Path) -> None:
    stream = LiveReceiptStream(sample_ledger, poll_interval=0.01)
    envelopes = list(stream.follow(max_items=2, timeout=0.1))
    assert [env.uuid for env in envelopes] == ["receipt-1", "receipt-2"]


def test_lab_client_uses_transport(sample_ledger: Path) -> None:
    stream = LiveReceiptStream(sample_ledger, poll_interval=0.01)
    published: list[tuple[str, bytes, dict[str, str]]] = []

    def transport(endpoint: str, payload: bytes, headers: dict[str, str]) -> None:
        published.append((endpoint, payload, headers))

    client = AISkillsLabClient("https://lab.tessrax.dev/ingest", transport=transport)
    delivered = client.stream_and_publish(stream, limit=1, timeout=0.1)
    assert delivered == 1
    assert published[0][0] == "https://lab.tessrax.dev/ingest"
    body = json.loads(published[0][1])
    assert body["uuid"] == "receipt-1"
    assert "Authorization" not in published[0][2]


def test_lab_client_requires_https(sample_ledger: Path) -> None:
    with pytest.raises(ValueError):
        AISkillsLabClient("http://lab.tessrax.dev")
