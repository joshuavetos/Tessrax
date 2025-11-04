"""Safe Tessrax SDK client with audited network operations.

All operations comply with Tessrax Governance Kernel v16 and enforce the
clauses ["AEP-001","POST-AUDIT-001","RVC-001","EAC-001"].  The client uses
standard library networking to avoid speculative dependencies and applies
runtime validation, retries, and integrity assertions.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict

BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 5
MAX_RETRIES = 3


class ConnectionError(RuntimeError):
    """Raised when the Tessrax SDK cannot reach the server."""


class SchemaError(RuntimeError):
    """Raised when the Tessrax SDK receives malformed JSON."""


class ProtocolError(RuntimeError):
    """Raised when the Tessrax server responds with an unexpected payload."""


@dataclass
class TessraxClient:
    """SDK facade that wraps the governed Tessrax HTTP endpoints."""

    base_url: str = BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    retries: int = MAX_RETRIES

    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            print(f"[TessraxSDK] {method} {url} attempt {attempt}/{self.retries}")
            request = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                    if not body:
                        return None
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise SchemaError(f"Invalid JSON from {url}") from exc
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise ProtocolError(f"HTTP {exc.code}: {exc.reason}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc
            time.sleep(0.5 * attempt)
        raise ConnectionError(f"Failed to reach {url}") from last_error

    def submit_claim(
        self,
        subject: str,
        metric: str,
        value: float,
        timestamp: str,
        source: str,
    ) -> Any:
        """Submit a structured claim to the Tessrax governance service."""

        if not all([subject, metric, source, timestamp]):
            raise ProtocolError("subject, metric, source, and timestamp must be provided")
        claim_id = sha256(f"{subject}:{metric}:{timestamp}".encode("utf-8")).hexdigest()[:16]
        payload = [
            {
                "claim_id": claim_id,
                "subject": subject,
                "metric": metric,
                "value": value,
                "unit": "unitless",
                "timestamp": timestamp,
                "source": source,
                "context": {},
            }
        ]
        return self._request("POST", "/claims", payload)

    def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Validate that a receipt exists in the Tessrax ledger endpoint."""

        if not receipt_id:
            raise ProtocolError("receipt_id must be supplied")
        ledger_entries = self._request("GET", "/ledger")
        if not isinstance(ledger_entries, list):
            raise SchemaError("Ledger endpoint did not return a list")
        for entry in ledger_entries:
            if isinstance(entry, dict) and entry.get("receipt_id") == receipt_id:
                return entry
        raise ProtocolError(f"Receipt {receipt_id} not found")


def submit_claim(subject: str, metric: str, value: float, timestamp: str, source: str) -> Any:
    """Convenience wrapper around :class:`TessraxClient.submit_claim`."""

    client = TessraxClient()
    return client.submit_claim(subject, metric, value, timestamp, source)


def verify_receipt(receipt_id: str) -> dict[str, Any]:
    """Convenience wrapper around :class:`TessraxClient.verify_receipt`."""

    client = TessraxClient()
    return client.verify_receipt(receipt_id)
