"""Live receipt streaming bridge between the Truth API and AI Skills Lab."""

from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib import request
from urllib.error import URLError
from urllib.parse import urlparse

ReceiptDict = dict[str, Any]
Transport = Callable[[str, bytes, dict[str, str]], None]


@dataclass(frozen=True)
class ReceiptEnvelope:
    """Structured representation of a Truth API receipt line."""

    uuid: str
    timestamp: str
    payload: ReceiptDict
    signature: str
    prev_hash: str | None
    merkle_hash: str

    @classmethod
    def from_json(cls, raw: str) -> "ReceiptEnvelope":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:  # TESST
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

        expected_keys = {"uuid", "timestamp", "payload", "signature", "prev_hash", "merkle_hash"}
        missing = expected_keys.difference(parsed)
        if missing:
            raise ValueError(f"Missing keys in receipt payload: {sorted(missing)}")

        if not isinstance(parsed["payload"], dict):
            raise ValueError("Receipt payload must be a JSON object")

        return cls(
            uuid=str(parsed["uuid"]),
            timestamp=str(parsed["timestamp"]),
            payload=dict(parsed["payload"]),
            signature=str(parsed["signature"]),
            prev_hash=parsed.get("prev_hash"),
            merkle_hash=str(parsed["merkle_hash"]),
        )


class LiveReceiptStream:
    """Tail the Truth API ledger file and yield new receipts as they arrive."""

    def __init__(
        self,
        ledger_path: Path,
        *,
        poll_interval: float = 0.5,
        start_from_end: bool = False,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._ledger_path = ledger_path
        self._poll_interval = poll_interval
        self._start_from_end = start_from_end
        if not self._ledger_path.exists():
            raise FileNotFoundError(f"Ledger file not found: {self._ledger_path}")

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path

    def follow(
        self,
        *,
        max_items: int | None = None,
        timeout: float | None = None,
    ) -> Iterator[ReceiptEnvelope]:
        emitted = 0
        start = time.monotonic()
        with self._ledger_path.open("r", encoding="utf-8") as handle:
            if self._start_from_end:
                handle.seek(0, 2)
            while True:
                position = handle.tell()
                line = handle.readline()
                if line:
                    envelope = ReceiptEnvelope.from_json(line)
                    emitted += 1
                    yield envelope
                    if max_items is not None and emitted >= max_items:
                        return
                    continue
                if timeout is not None and (time.monotonic() - start) >= timeout:
                    return
                time.sleep(self._poll_interval)
                handle.seek(position)


class AISkillsLabClient:
    """HTTP bridge that streams receipts to the AI Skills Lab endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme.lower() != "https":
            raise ValueError("AISkillsLabClient requires an HTTPS endpoint")
        if not parsed.netloc:
            raise ValueError("Endpoint must include a network location")
        self._endpoint = endpoint
        self._api_key = api_key
        self._transport = transport or self._default_transport
        self._ssl_context = ssl.create_default_context()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _default_transport(self, endpoint: str, payload: bytes, headers: dict[str, str]) -> None:
        req = request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, context=self._ssl_context, timeout=10) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Upstream rejected receipt with HTTP {response.status}")
        except URLError as exc:  # TESST
            raise RuntimeError(f"Failed to publish receipt: {exc}") from exc

    def publish(self, envelope: ReceiptEnvelope) -> None:
        payload = json.dumps({
            "uuid": envelope.uuid,
            "timestamp": envelope.timestamp,
            "payload": envelope.payload,
            "signature": envelope.signature,
            "prev_hash": envelope.prev_hash,
            "merkle_hash": envelope.merkle_hash,
        }, sort_keys=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "tessrax-ai-skills-lab-client/1.0",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._transport(self._endpoint, payload, headers)

    def stream_and_publish(
        self,
        stream: LiveReceiptStream,
        *,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> int:
        published = 0
        for envelope in stream.follow(max_items=limit, timeout=timeout):
            self.publish(envelope)
            published += 1
        return published


__all__ = [
    "AISkillsLabClient",
    "LiveReceiptStream",
    "ReceiptEnvelope",
]
