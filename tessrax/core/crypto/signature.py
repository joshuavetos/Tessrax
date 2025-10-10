"""
Ed25519 signature utilities for Tessrax receipts.
Requires 'pynacl' (pip install pynacl)
"""

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
import json, hashlib

def new_keypair():
    sk = SigningKey.generate()
    vk = sk.verify_key
    return sk.encode(encoder=HexEncoder).decode(), vk.encode(encoder=HexEncoder).decode()

def sign_receipt(receipt:dict, sk_hex:str)->dict:
    sk = SigningKey(sk_hex, encoder=HexEncoder)
    msg = json.dumps(receipt, sort_keys=True).encode()
    sig = sk.sign(msg).signature.hex()
    receipt["signature"] = sig
    return receipt

def verify_receipt(receipt:dict, vk_hex:str)->bool:
    vk = VerifyKey(vk_hex, encoder=HexEncoder)
    msg = json.dumps({k:v for k,v in receipt.items() if k!="signature"}, sort_keys=True).encode()
    try:
        vk.verify(msg, bytes.fromhex(receipt["signature"]))
        return True
    except Exception:
        return False