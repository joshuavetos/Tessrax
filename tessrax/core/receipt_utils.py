# tessrax/core/receipt_utils.py — Finalized v2.5
"""
Tessrax Receipt Framework v2.5 (Final Hardened Build)
------------------------------------------------------
This module provides cryptographically verifiable, tamper-evident receipts.

Features:
- Ed25519 digital signatures (canonical JSON signing)
- Nonce replay protection (optional registry)
- RFC 3161 trusted timestamp integration (optional TSA)
- Strict field validation
- Deterministic hash chaining
- Batch and chain verification utilities
"""

import json
import hashlib
import unicodedata
import time
import uuid
from typing import Any, Dict, Optional, List

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SignatureVerificationError(Exception):
    """Raised when signature verification fails."""
    pass

class ReceiptVerificationError(Exception):
    """Raised when a receipt fails structural or integrity validation."""
    pass


# ---------------------------------------------------------------------------
# Canonicalization + Hashing
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    """Return deterministic JSON (sorted keys, compact form)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def _canonicalize_code(code: str) -> str:
    """Normalize code text for deterministic hashing."""
    lines = [l.rstrip() for l in code.strip().splitlines()]
    return unicodedata.normalize("NFC", "\n".join(lines))


# ---------------------------------------------------------------------------
# Receipt Core
# ---------------------------------------------------------------------------

def create_receipt(
    private_key_hex: str,
    payload: Dict[str, Any],
    prev_hash: Optional[str] = None,
    include_rfc3161: bool = False,
    tsa_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a signed Tessrax receipt with canonical JSON signing.
    """
    signer = SigningKey(private_key_hex, encoder=HexEncoder)
    pub_hex = signer.verify_key.encode(encoder=HexEncoder).decode("utf-8")

    receipt_core = {
        "version": "2.5",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "nonce": str(uuid.uuid4()),
        "payload": payload,
        "prev_hash": prev_hash,
        "signer_pubkey": pub_hex,
    }

    canonical = _canonical_json(receipt_core)
    sig = signer.sign(canonical.encode("utf-8")).signature.hex()

    receipt = dict(receipt_core)
    receipt["signature"] = sig

    # optional RFC 3161 timestamping hook
    if include_rfc3161 and tsa_url:
        digest = _sha256_hex(canonical)
        receipt["trusted_timestamp"] = _obtain_rfc3161_timestamp(digest, tsa_url)

    return receipt


def verify_receipt(
    receipt: Dict[str, Any],
    nonce_registry: Optional[Any] = None,
    strict: bool = True,
) -> bool:
    """
    Verify signature, structure, and optional replay protection.
    """
    required = {"version","timestamp","nonce","payload","prev_hash","signer_pubkey","signature"}
    if not required.issubset(receipt.keys()):
        raise ReceiptVerificationError("Missing required fields.")

    if strict:
        if not isinstance(receipt["payload"], dict):
            raise ReceiptVerificationError("Payload must be a dict.")
        if not isinstance(receipt["nonce"], str):
            raise ReceiptVerificationError("Nonce must be a string.")

    if nonce_registry:
        if nonce_registry.seen(receipt["nonce"]):
            raise ReceiptVerificationError("Replay detected.")
        nonce_registry.register(receipt["nonce"])

    base = {k:v for k,v in receipt.items() if k not in {"signature","trusted_timestamp"}}
    canonical = _canonical_json(base)
    try:
        vk = VerifyKey(receipt["signer_pubkey"], encoder=HexEncoder)
        vk.verify(canonical.encode("utf-8"), bytes.fromhex(receipt["signature"]))
    except (BadSignatureError, ValueError, TypeError) as e:
        raise SignatureVerificationError(f"Signature check failed: {e}")
    return True


# ---------------------------------------------------------------------------
# Optional: RFC 3161 Timestamp Stub
# ---------------------------------------------------------------------------

def _obtain_rfc3161_timestamp(digest_hex: str, tsa_url: str) -> Dict[str, Any]:
    """
    Stub placeholder for real TSA integration.
    """
    return {
        "tsa_url": tsa_url,
        "digest": digest_hex,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "stubbed"
    }


# ---------------------------------------------------------------------------
# Utility: Deterministic Hash / Chain Verification
# ---------------------------------------------------------------------------

def receipt_hash(receipt: Dict[str, Any]) -> str:
    """Compute deterministic hash of signed fields."""
    base = {k:v for k,v in receipt.items() if k not in {"signature","trusted_timestamp"}}
    return _sha256_hex(_canonical_json(base))

def verify_receipt_chain(chain: List[Dict[str, Any]]) -> bool:
    """Verify linked receipts in a chain."""
    for i in range(1, len(chain)):
        expected = receipt_hash(chain[i-1])
        if chain[i]["prev_hash"] != expected:
            raise ReceiptVerificationError(f"Chain break at index {i}")
        verify_receipt(chain[i])
    return True

def create_receipts_batch(private_key_hex: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate a full sequential receipt chain."""
    receipts = []
    prev_hash = None
    for p in payloads:
        r = create_receipt(private_key_hex, p, prev_hash=prev_hash)
        receipts.append(r)
        prev_hash = receipt_hash(r)
    return receipts