"""Deterministic serialization utilities for TGP payloads."""
from __future__ import annotations

import json
import hashlib
from typing import Any

import cbor2


def encode_json(payload: Any) -> str:
    """Serialize payload to canonical JSON with sorted keys."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_json(serialized: str) -> Any:
    """Deserialize JSON produced by :func:`encode_json`."""
    return json.loads(serialized)


def encode_cbor(payload: Any) -> bytes:
    """Serialize payload using canonical CBOR encoding."""
    return cbor2.dumps(payload, canonical=True)


def decode_cbor(blob: bytes) -> Any:
    """Deserialize CBOR produced by :func:`encode_cbor`."""
    return cbor2.loads(blob)


def hash_payload(payload: Any) -> str:
    """Return a SHA-256 digest for a payload using canonical JSON encoding."""
    canonical = encode_json(payload).encode()
    return hashlib.sha256(canonical).hexdigest()


def ensure_roundtrip(payload: Any) -> None:
    """Runtime verification ensuring JSON/CBOR round-trips consistently."""
    json_encoded = encode_json(payload)
    json_decoded = decode_json(json_encoded)
    cbor_encoded = encode_cbor(payload)
    cbor_decoded = decode_cbor(cbor_encoded)
    if json_decoded != cbor_decoded:
        raise ValueError("TESST violation: JSON and CBOR decodings diverge")


__all__ = [
    "encode_json",
    "decode_json",
    "encode_cbor",
    "decode_cbor",
    "hash_payload",
    "ensure_roundtrip",
]
