#!/bin/bash
# Tessrax Session Hash and Signature
# Summarizes all ledger entries and signs the session hash with Ed25519.

LEDGER_DIR="ledger"
HASH_FILE="$LEDGER_DIR/session.hash"
SIGNED_FILE="$LEDGER_DIR/session.signed.json"

echo "🔐 Generating session hash..."
find "$LEDGER_DIR" -type f -name "*.json" -exec sha256sum {} \; | sort -k 2 > "$HASH_FILE"

echo "🖋️ Signing session hash..."
python3 - <<'PYCODE'
import json, hashlib
from nacl import signing, encoding

# Load private key
with open("tessrax_private.key", "rb") as f:
    sk = signing.SigningKey(f.read())
vk = sk.verify_key

# Read session hash
with open("ledger/session.hash","rb") as f:
    data = f.read()

sig = sk.sign(data).signature.hex()
record = {
    "session_hash": hashlib.sha256(data).hexdigest(),
    "signature": f"ed25519:{sig}",
    "public_key": vk.encode(encoder=encoding.HexEncoder).decode()
}
with open("ledger/session.signed.json","w") as out:
    json.dump(record,out,indent=2)
print("✅ Session signed successfully.")
PYCODE