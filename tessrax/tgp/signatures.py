"""Signature utilities for the Tessrax Governance Protocol."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from nacl import encoding, signing

from .cbor_codec import hash_payload


@dataclass(frozen=True)
class SignatureEnvelope:
    """Encapsulates signed payload metadata."""

    payload_hash: str
    signature: str
    public_key: str
    nonce: str
    issued_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload_hash": self.payload_hash,
            "signature": self.signature,
            "public_key": self.public_key,
            "nonce": self.nonce,
            "issued_at": self.issued_at,
        }


class ReplayProtection:
    """Tracks nonces to guard against signature replay attacks."""

    def __init__(self) -> None:
        self._seen: set[Tuple[str, str]] = set()

    def register(self, public_key: str, nonce: str) -> None:
        key = (public_key, nonce)
        if key in self._seen:
            raise ValueError("Replay detected for nonce")
        self._seen.add(key)


class Signer:
    """Helper wrapper around an Ed25519 signing key."""

    def __init__(self, secret_key: bytes) -> None:
        if len(secret_key) != 32:
            raise ValueError("Ed25519 secret keys must be 32 bytes")
        self._signing_key = signing.SigningKey(secret_key)

    @classmethod
    def from_hex(cls, secret_key_hex: str) -> "Signer":
        return cls(encoding.HexEncoder.decode(secret_key_hex))

    @property
    def public_key(self) -> str:
        verify_key = self._signing_key.verify_key
        return verify_key.encode(encoder=encoding.HexEncoder).decode()

    def sign(self, payload: Dict[str, Any], nonce: str, issued_at: Optional[float] = None) -> SignatureEnvelope:
        issued_at = time.time() if issued_at is None else issued_at
        digest = hash_payload(payload)
        message = json.dumps(
            {
                "payload_hash": digest,
                "nonce": nonce,
                "issued_at": issued_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        signature = self._signing_key.sign(message).signature
        signature_hex = encoding.HexEncoder.encode(signature).decode()
        return SignatureEnvelope(
            payload_hash=digest,
            signature=signature_hex,
            public_key=self.public_key,
            nonce=nonce,
            issued_at=issued_at,
        )


class SignatureVerifier:
    """Verifies signature envelopes with replay protection."""

    def __init__(self, replay_protection: ReplayProtection | None = None) -> None:
        self._replay = replay_protection or ReplayProtection()

    def verify(self, envelope: SignatureEnvelope, payload: Dict[str, Any]) -> bool:
        if envelope.payload_hash != hash_payload(payload):
            raise ValueError("Payload hash mismatch during verification")
        message = json.dumps(
            {
                "payload_hash": envelope.payload_hash,
                "nonce": envelope.nonce,
                "issued_at": envelope.issued_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        verify_key = signing.VerifyKey(envelope.public_key, encoder=encoding.HexEncoder)
        signature_bytes = encoding.HexEncoder.decode(envelope.signature)
        verify_key.verify(message, signature_bytes)
        self._replay.register(envelope.public_key, envelope.nonce)
        return True


__all__ = [
    "SignatureEnvelope",
    "ReplayProtection",
    "Signer",
    "SignatureVerifier",
]
