"""
tessrax/core/receipts.py
------------------------
Tessrax Receipt System (Hardened v2.1)

Provides creation, verification, and key management utilities
for tamper-evident receipts used across Tessrax modules.

Core Features:
✓ Deterministic JSON encoding for reproducibility
✓ Ed25519 signing / verification (PyNaCl)
✓ Provenance + integrity tracking
✓ Safe error handling for audit logs
"""

import json
import time
import hashlib
from typing import Any, Dict, Tuple

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder


# ------------------------------------------------------------
# Utility Helpers
# ------------------------------------------------------------
def _canonical_json(data: Any) -> str:
    """Deterministically encode JSON with sorted keys."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(s: str) -> str:
    """Return SHA256 hex digest."""
    return hashlib.sha256(s.encode()).hexdigest()


# ------------------------------------------------------------
# Key Utilities
# ------------------------------------------------------------
def generate_keypair() -> Tuple[str, str]:
    """
    Generates a new Ed25519 keypair.
    Returns (private_key_hex, public_key_hex).
    """
    sk = SigningKey.generate()
    pk = sk.verify_key
    return (
        sk.encode(encoder=HexEncoder).decode(),
        pk.encode(encoder=HexEncoder).decode(),
    )


def load_keys(private_hex: str) -> Tuple[SigningKey, VerifyKey]:
    """Load signing and verify keys from a hex private key."""
    sk = SigningKey(private_hex, encoder=HexEncoder)
    vk = sk.verify_key
    return sk, vk


# ------------------------------------------------------------
# Receipt Creation
# ------------------------------------------------------------
def create_receipt(private_key_hex: str, event_payload: Dict[str, Any], executor_id: str = "unknown") -> Dict[str, Any]:
    """
    Create a signed, tamper-evident receipt for an event.

    The payload is hashed canonically, signed, and accompanied
    by provenance and integrity fields.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    canonical_payload = _canonical_json(event_payload)
    payload_hash = _sha256(canonical_payload)

    signing_key = SigningKey(private_key_hex, encoder=HexEncoder)
    signature = signing_key.sign(payload_hash.encode()).signature.hex()
    verify_key_hex = signing_key.verify_key.encode(encoder=HexEncoder).decode()

    integrity = _sha256(payload_hash + signature + executor_id)

    return {
        "timestamp": timestamp,
        "executor_id": executor_id,
        "payload": event_payload,
        "payload_hash": payload_hash,
        "signature": signature,
        "verify_key": verify_key_hex,
        "integrity_hash": integrity,
    }


# ------------------------------------------------------------
# Receipt Verification
# ------------------------------------------------------------
def verify_receipt(receipt: Dict[str, Any]) -> bool:
    """
    Verify authenticity and integrity of a Tessrax receipt.
    Returns True if valid, raises ValueError if tampering detected.
    """
    try:
        # Step 1: Recalculate payload hash
        recalculated_hash = _sha256(_canonical_json(receipt["payload"]))
        if recalculated_hash != receipt.get("payload_hash"):
            raise ValueError("Payload hash mismatch")

        # Step 2: Verify cryptographic signature
        verify_key = VerifyKey(receipt["verify_key"], encoder=HexEncoder)
        verify_key.verify(
            receipt["payload_hash"].encode(),
            bytes.fromhex(receipt["signature"]),
        )

        # Step 3: Check integrity
        expected_integrity = _sha256(
            receipt["payload_hash"] + receipt["signature"] + receipt["executor_id"]
        )
        if expected_integrity != receipt.get("integrity_hash"):
            raise ValueError("Integrity hash mismatch")

        return True

    except Exception as e:
        print(f"[Receipt Verification Failed] {e}")
        return False


# ------------------------------------------------------------
# Manual Test
# ------------------------------------------------------------
if __name__ == "__main__":
    priv, pub = generate_keypair()
    event = {"action": "demo_test", "value": 42}

    receipt = create_receipt(priv, event, executor_id="demo_agent")
    print(json.dumps(receipt, indent=2))
    print("Verification:", verify_receipt(receipt))