"""
tessrax/core/receipt_utils.py
-----------------------------
Cryptographically verifiable receipts for Tessrax operations.

✓ Deterministic JSON encoding
✓ Provenance & executor tracking
✓ NaCl Ed25519 signing / verification
✓ Integrity hash for all payloads
"""

import json
import time
import hashlib
from typing import Any, Dict

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _canonical_json(data: Any) -> str:
    """Serialize with deterministic ordering for reproducible hashes."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(data: str) -> str:
    """Return SHA-256 hex digest for a given string."""
    return hashlib.sha256(data.encode()).hexdigest()


# ------------------------------------------------------------
# Receipt Creation
# ------------------------------------------------------------
def create_receipt(private_key_hex: str, event_payload: Dict[str, Any], executor_id: str = "unknown") -> Dict[str, Any]:
    """
    Create a signed, tamper-evident receipt from an event payload.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Canonical JSON for integrity
    canonical_payload = _canonical_json(event_payload)
    payload_hash = _sha256(canonical_payload)

    # Sign the payload hash
    signing_key = SigningKey(private_key_hex, encoder=HexEncoder)
    signature = signing_key.sign(payload_hash.encode(), encoder=HexEncoder).signature.decode()

    receipt = {
        "timestamp": timestamp,
        "executor_id": executor_id,
        "payload": event_payload,
        "payload_hash": payload_hash,
        "signature": signature,
        "verify_key": signing_key.verify_key.encode(encoder=HexEncoder).decode(),
        "integrity_hash": _sha256(payload_hash + signature + executor_id)
    }
    return receipt


# ------------------------------------------------------------
# Receipt Verification
# ------------------------------------------------------------
def verify_receipt(receipt: Dict[str, Any]) -> bool:
    """
    Verify the authenticity and integrity of a receipt.
    """
    try:
        payload_hash = _sha256(_canonical_json(receipt["payload"]))
        if payload_hash != receipt.get("payload_hash"):
            raise ValueError("Payload hash mismatch")

        verify_key = VerifyKey(receipt["verify_key"], encoder=HexEncoder)
        verify_key.verify(receipt["payload_hash"].encode(), bytes.fromhex(receipt["signature"]))

        expected_integrity = _sha256(
            receipt["payload_hash"] + receipt["signature"] + receipt["executor_id"]
        )
        if expected_integrity != receipt["integrity_hash"]:
            raise ValueError("Integrity hash mismatch")

        return True

    except Exception as e:
        print(f"[Receipt Verification Failed] {e}")
        return False


# ------------------------------------------------------------
# Example (manual test)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Generate a test key
    sk = SigningKey.generate()
    pk = sk.verify_key

    private_hex = sk.encode(encoder=HexEncoder).decode()
    payload = {"action": "test", "value": 42}

    # Create and verify
    receipt = create_receipt(private_hex, payload, executor_id="demo_agent")
    print(json.dumps(receipt, indent=2))
    print("Verification result:", verify_receipt(receipt))