"""
Tessrax Ledger Security Upgrade
Implements Ed25519 signing and verification for immutable ledger entries.
"""

import hashlib, json, datetime
from nacl import signing, encoding

# Load or create keys
def load_or_create_keys():
    try:
        with open("tessrax_private.key", "rb") as f:
            private_key = signing.SigningKey(f.read())
    except FileNotFoundError:
        private_key = signing.SigningKey.generate()
        with open("tessrax_private.key", "wb") as f:
            f.write(private_key.encode())
        with open("tessrax_public.key", "wb") as f:
            f.write(private_key.verify_key.encode())
    return private_key, private_key.verify_key

private_key, public_key = load_or_create_keys()

def sign_ledger_entry(entry: dict) -> dict:
    """Return entry with hash + Ed25519 signature."""
    entry["timestamp"] = datetime.datetime.utcnow().isoformat()
    entry_json = json.dumps(entry, sort_keys=True).encode("utf-8")
    entry_hash = hashlib.sha256(entry_json).hexdigest()
    signature = private_key.sign(entry_json).signature.hex()
    return {
        "entry": entry,
        "hash": f"sha256:{entry_hash}",
        "signature": f"ed25519:{signature}",
        "public_key": public_key.encode(encoder=encoding.HexEncoder).decode()
    }

def verify_entry(signed_entry: dict) -> bool:
    """Verify signature using stored public key."""
    verify_key = signing.VerifyKey(
        signed_entry["public_key"], encoder=encoding.HexEncoder
    )
    message = json.dumps(signed_entry["entry"], sort_keys=True).encode("utf-8")
    try:
        verify_key.verify(message, bytes.fromhex(signed_entry["signature"].split(":")[1]))
        return True
    except Exception:
        return False

if __name__ == "__main__":
    # Example use
    entry = {"claim_a": "Company will be carbon neutral by 2030.", "claim_b": "Company increased emissions 5% in 2024."}
    signed = sign_ledger_entry(entry)
    print(json.dumps(signed, indent=2))
    print("Verification:", verify_entry(signed))