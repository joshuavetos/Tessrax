"""
Automates periodic Ed25519 key rotation and updates revoked.json.
"""
from nacl import signing, encoding
import json, datetime, os

ROTATION_DAYS = 90

def rotate_keys():
    new_key = signing.SigningKey.generate()
    timestamp = datetime.datetime.utcnow().isoformat()
    os.rename("tessrax_private.key", f"archived/private_{timestamp}.key")
    os.rename("tessrax_public.key", f"archived/public_{timestamp}.key")

    with open("tessrax_private.key", "wb") as f:
        f.write(new_key.encode())
    with open("tessrax_public.key", "wb") as f:
        f.write(new_key.verify_key.encode())

    with open("keys/revoked.json", "r+") as f:
        revoked = json.load(f)
        revoked.append({"revoked_at": timestamp, "reason": "rotated"})
        f.seek(0)
        json.dump(revoked, f, indent=2)
    print("✅ Keys rotated and old public key revoked.")

if __name__ == "__main__":
    rotate_keys()