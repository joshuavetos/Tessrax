"""
Lightweight key vault for Tessrax signatures.
Stores and rotates Ed25519 keypairs locally.
"""

import os, json
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder

KEY_FILE = os.path.expanduser("~/.tessrax_keys.json")

def init_keys():
    """Create new keypair if none exists."""
    if not os.path.exists(KEY_FILE):
        sk = SigningKey.generate()
        vk = sk.verify_key
        data = {
            "private_key": sk.encode(encoder=HexEncoder).decode(),
            "public_key": vk.encode(encoder=HexEncoder).decode()
        }
        with open(KEY_FILE, "w") as f: json.dump(data, f)
        print("New Tessrax keypair generated.")
    else:
        print("Keypair already exists.")

def load_keys():
    with open(KEY_FILE) as f: return json.load(f)

def rotate_keys():
    """Generate new pair and archive old one."""
    old = load_keys()
    new_sk = SigningKey.generate()
    new_vk = new_sk.verify_key
    archive = KEY_FILE + ".bak"
    os.replace(KEY_FILE, archive)
    with open(KEY_FILE, "w") as f:
        json.dump({
            "private_key": new_sk.encode(encoder=HexEncoder).decode(),
            "public_key": new_vk.encode(encoder=HexEncoder).decode()
        }, f)
    print(f"Rotated keys. Old pair archived at {archive}")