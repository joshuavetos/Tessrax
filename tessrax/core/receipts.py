"""
tessrax/core/receipts.py
------------------------
Tessrax Receipt System v3.0 — Unified & Hardened

Features:
✓ Deterministic JSON encoding for reproducible hashing
✓ Ed25519 signing & verification (PyNaCl)
✓ Nonce registry to prevent replay attacks
✓ Revocation registry to manage revoked keys or receipts
✓ Integrity hash to detect tampering
✓ Graceful error handling and auditable verification logs
"""

import json
import time
import hashlib
from typing import Any, Dict, Tuple, Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder


# ============================================================
# Utility Helpers
# ============================================================
def _canonical_json(data: Any) -> str:
    """Return deterministic JSON string (sorted keys, no spaces)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _sha256(s: str) -> str:
    """Return SHA-256 hex digest."""
    return hashlib.sha256(s.encode()).hexdigest()


# ============================================================
# Key Utilities
# ============================================================
def generate_keypair() -> Tuple[str, str]:
    """
    Generate a new Ed25519 keypair.
    Returns (private_key_hex, public_key_hex).
    """
    sk = SigningKey.generate()
    return (
        sk.encode(encoder=HexEncoder).decode(),
        sk.verify_key.encode(encoder=HexEncoder).decode(),
    )


def load_keys(private_key_hex: str) -> Tuple[SigningKey, VerifyKey]:
    """
    Load signing and verify keys from a hex private key.
    """
    sk = SigningKey(private_key_hex, encoder=HexEncoder)
    return sk, sk.verify_key


# ============================================================
# Registries
# ============================================================
class NonceRegistry:
    """
    Simple in-memory nonce registry to prevent replay attacks.
    """

    def __init__(self):
        self._nonces = set()

    def register(self, nonce: str) -> bool:
        """
        Register a nonce. Returns False if it was already used.
        """
        if nonce in self._nonces:
            return False
        self._nonces.add(nonce)
        return True

    def exists(self, nonce: str) -> bool:
        """Check whether a nonce is already registered."""
        return nonce in self._nonces


class RevocationRegistry:
    """
    Track revoked keys or receipt IDs for auditing and blocking.
    """

    def __init__(self):
        self._revoked = set()

    def revoke(self, key: str) -> None:
        """Mark a key or receipt ID as revoked."""
        self._revoked.add(key)

    def is_revoked(self, key: str) -> bool:
        """Return True if a key or receipt ID has been revoked."""
        return key in self._revoked


# ============================================================
# Receipt Creation
# ============================================================
def create_receipt(
    private_key_hex: str,
    event_payload: Dict[str, Any],
    executor_id: str = "unknown",
    nonce: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a signed Tessrax receipt for a given payload.

    The receipt includes:
      - timestamp
      - canonical payload hash
      - Ed25519 signature & verify key
      - executor_id and optional nonce
      - integrity_hash for tamper evidence
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    canonical_payload = _canonical_json(event_payload)
    payload_hash = _sha256(canonical_payload)

    sk = SigningKey(private_key_hex, encoder=HexEncoder)
    signature_bytes = sk.sign(payload_hash.encode()).signature
    signature_hex = signature_bytes.hex()
    verify_key_hex = sk.verify_key.encode(encoder=HexEncoder).decode()

    integrity = _sha256(payload_hash + signature_hex + executor_id)

    receipt = {
        "timestamp": timestamp,
        "executor_id": executor_id,
        "nonce": nonce,
        "payload": event_payload,
        "payload_hash": payload_hash,
        "signature": signature_hex,
        "verify_key": verify_key_hex,
        "integrity_hash": integrity,
    }
    return receipt


# ============================================================
# Receipt Verification
# ============================================================
def verify_receipt(
    receipt: Dict[str, Any],
    *,
    nonce_registry: Optional[NonceRegistry] = None,
    revocation_registry: Optional[RevocationRegistry] = None,
    strict: bool = True,
) -> bool:
    """
    Verify authenticity and integrity of a Tessrax receipt.

    Steps:
      1. Recalculate payload hash from canonical JSON
      2. Check Ed25519 signature
      3. Verify integrity hash
      4. Validate nonce uniqueness (if registry provided)
      5. Check revocation registry

    Returns True if all checks pass, otherwise raises ValueError or returns False.
    """
    try:
        # Ensure required fields
        required = {"timestamp", "executor_id", "payload", "payload_hash", "signature", "verify_key"}
        if not all(k in receipt for k in required):
            raise ValueError("Missing required receipt fields")

        # 1. Recalculate payload hash
        recalculated_hash = _sha256(_canonical_json(receipt["payload"]))
        if recalculated_hash != receipt.get("payload_hash"):
            raise ValueError("Payload hash mismatch")

        # 2. Verify signature
        verify_key = VerifyKey(receipt["verify_key"], encoder=HexEncoder)
        verify_key.verify(
            receipt["payload_hash"].encode(),
            bytes.fromhex(receipt["signature"]),
        )

        # 3. Verify integrity hash
        expected_integrity = _sha256(
            receipt["payload_hash"] + receipt["signature"] + receipt["executor_id"]
        )
        if expected_integrity != receipt.get("integrity_hash"):
            raise ValueError("Integrity hash mismatch")

        # 4. Check nonce uniqueness if registry provided
        nonce = receipt.get("nonce")
        if nonce and nonce_registry:
            if not nonce_registry.register(nonce):
                raise ValueError("Nonce already used (possible replay attack)")

        # 5. Check revocation registry
        if revocation_registry:
            if revocation_registry.is_revoked(receipt.get("verify_key")):
                raise ValueError("Key has been revoked")

        return True

    except Exception as e:
        print(f"[Receipt Verification Failed] {e}")
        return False


# ============================================================
# Manual Test
# ============================================================
if __name__ == "__main__":
    priv, pub = generate_keypair()
    payload = {"action": "test_run", "value": 42}

    nonce_reg = NonceRegistry()
    revoc_reg = RevocationRegistry()

    receipt = create_receipt(priv, payload, executor_id="demo_agent", nonce="abc123")
    print("Generated Receipt:", json.dumps(receipt, indent=2))

    print("Verification (first time):", verify_receipt(receipt, nonce_registry=nonce_reg, revocation_registry=revoc_reg))
    print("Verification (replay attempt):", verify_receipt(receipt, nonce_registry=nonce_reg, revocation_registry=revoc_reg))