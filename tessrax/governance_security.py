"""Governance security utilities for signature validation and timestamp anchoring."""

from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

from .types import GovernanceDecision


def _json_dumps(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class TimestampToken:
    """Lightweight representation of an RFC-3161 style timestamp token."""

    issued_at: datetime
    nonce: str
    digest: str

    def encode(self) -> str:
        payload = {
            "issued_at": self.issued_at.isoformat(),
            "nonce": self.nonce,
            "digest": self.digest,
        }
        return base64.urlsafe_b64encode(_json_dumps(payload)).decode("ascii")

    @classmethod
    def decode(cls, token: str) -> "TimestampToken":
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        data = json.loads(raw)
        issued_at = datetime.fromisoformat(data["issued_at"])
        if issued_at.tzinfo is None:
            issued_at = issued_at.replace(tzinfo=timezone.utc)
        return cls(issued_at=issued_at, nonce=data["nonce"], digest=data["digest"])


class TimestampAuthority:
    """Create and validate timestamp tokens tied to canonical decision payloads."""

    def __init__(self, hash_name: str = "sha256") -> None:
        self._hash_name = hash_name

    def _digest(self, payload: Mapping[str, Any]) -> str:
        import hashlib

        return hashlib.new(self._hash_name, _json_dumps(payload)).hexdigest()

    def issue(self, payload: Mapping[str, Any]) -> TimestampToken:
        issued_at = datetime.now(timezone.utc)
        digest = self._digest(payload)
        nonce = secrets.token_hex(8)
        return TimestampToken(issued_at=issued_at, nonce=nonce, digest=digest)

    def verify(
        self,
        payload: Mapping[str, Any],
        token: TimestampToken,
        leeway: timedelta = timedelta(minutes=5),
    ) -> bool:
        if abs(datetime.now(timezone.utc) - token.issued_at) > leeway:
            return False
        expected = self._digest(payload)
        return secrets.compare_digest(expected, token.digest)


@dataclass(frozen=True)
class DecisionSignature:
    """Coupled digital signature and timestamp token."""

    signature: str
    timestamp_token: str


class SignatureAuthority:
    """Generate and verify deterministic HMAC-based decision signatures."""

    def __init__(
        self,
        secret: str,
        hash_name: str = "sha256",
        *,
        timestamp_authority: Optional[TimestampAuthority] = None,
    ) -> None:
        self._secret = secret.encode("utf-8")
        self._hash_name = hash_name
        self._timestamp = timestamp_authority or TimestampAuthority(hash_name=hash_name)

    def _payload_bytes(self, payload: Mapping[str, Any], token: TimestampToken) -> bytes:
        combined = {"payload": payload, "timestamp_token": token.encode()}
        return _json_dumps(combined)

    def sign(self, decision: GovernanceDecision) -> DecisionSignature:
        payload = decision.canonical_document()
        token = self._timestamp.issue(payload)
        message = self._payload_bytes(payload, token)
        import hmac

        digest = hmac.new(self._secret, message, self._hash_name)
        signature = base64.urlsafe_b64encode(digest.digest()).decode("ascii")
        return DecisionSignature(signature=signature, timestamp_token=token.encode())

    def verify(self, decision: GovernanceDecision, signature: DecisionSignature) -> bool:
        try:
            token = TimestampToken.decode(signature.timestamp_token)
        except Exception:  # pragma: no cover - defensive decoding guard
            return False
        payload = decision.canonical_document()
        if not self._timestamp.verify(payload, token):
            return False
        import hmac

        message = self._payload_bytes(payload, token)
        digest = hmac.new(self._secret, message, self._hash_name)
        expected_signature = base64.urlsafe_b64encode(digest.digest()).decode("ascii")
        return secrets.compare_digest(expected_signature, signature.signature)
