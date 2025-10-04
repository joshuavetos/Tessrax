"""
Tessrax Receipts Module v4.1 — cryptographically verifiable, replay-proof,
and legally auditable execution receipts.

Features:
 • Ed25519 (Hex) signatures with strict verification mode
 • Canonical JSON + Unicode-normalized code hashing
 • Persistent nonce registry (SQLite WAL) — replay immunity
 • Payload-hash revocation registry — surgical control
 • Optional RFC 3161 timestamp authority (TSA) integration
 • Structured exceptions for audit logging
 • Batch verification + serialization helpers
 • Graceful TSA fallback (require_tsa flag)
"""

import hashlib, json, time, uuid, unicodedata, sqlite3, certifi, threading
from pathlib import Path
from typing import Any, Dict, Optional, List

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from nacl.exceptions import BadSignatureError

# Optional RFC 3161 timestamp support
try:
    from rfc3161_client import TimestampRequestBuilder, decode_timestamp_response
    import requests
    RFC3161_AVAILABLE = True
except ImportError:
    RFC3161_AVAILABLE = False


# =========================================================================
# Exceptions
# =========================================================================

class ReceiptError(Exception):
    """Base class for all receipt-related errors."""

class ReceiptVerificationError(ReceiptError):
    """Raised when a receipt fails verification."""
    def __init__(self, field:str, expected:Any, actual:Any, msg:str):
        super().__init__(msg)
        self.field, self.expected, self.actual = field, expected, actual
    def to_json(self)->Dict[str,Any]:
        return {
            "field":self.field,"expected":self.expected,
            "actual":self.actual,"message":str(self)
        }


# =========================================================================
# Canonicalization Helpers
# =========================================================================

def _canonicalize_code(code:str)->str:
    """Normalize code text: trim, tabs→spaces, normalize Unicode."""
    lines=[l.rstrip() for l in code.strip().splitlines()]
    text="\n".join(lines).replace("\t","    ")
    return unicodedata.normalize("NFC",text)

def _canonical_json(obj:Any)->str:
    """Deterministic JSON serializer (sort keys, no spaces)."""
    return json.dumps(obj,sort_keys=True,separators=(",",":"))

def _sha256_hex(data:str)->str:
    """SHA-256 hex digest."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =========================================================================
# Persistent Registries (SQLite-backed)
# =========================================================================

class SQLiteRegistry:
    """Lightweight SQLite key-value store (used by nonce & revocation registries)."""
    def __init__(self,path:Path):
        self.lock=threading.Lock()
        self.conn=sqlite3.connect(path,check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("CREATE TABLE IF NOT EXISTS registry(key TEXT PRIMARY KEY,value TEXT,timestamp REAL DEFAULT (strftime('%s','now')));")
    def has(self,key:str)->bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM registry WHERE key=?;",(key,)).fetchone() is not None
    def add(self,key:str,value:str="1")->bool:
        with self.lock:
            try:
                self.conn.execute("INSERT INTO registry(key,value) VALUES(?,?);",(key,value))
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

class NonceRegistry(SQLiteRegistry):
    """Persistent replay-protection registry."""
    def register(self,nonce:str)->bool: return self.add(nonce)
    def seen(self,nonce:str)->bool: return self.has(nonce)
    def prune_old(self,days:int=90):
        cutoff=time.time()-(days*86400)
        with self.lock:
            self.conn.execute("DELETE FROM registry WHERE timestamp<?;",(cutoff,))
            self.conn.commit()

class RevocationRegistry(SQLiteRegistry):
    """Persistent payload-revocation registry."""
    def revoke(self,payload_hash:str): self.add(payload_hash)
    def is_revoked(self,payload_hash:str)->bool: return self.has(payload_hash)
    def export_revocations(self)->List[str]:
        return [r[0] for r in self.conn.execute("SELECT key FROM registry;")]
    def import_revocations(self,hashes:List[str]):
        for h in hashes: self.revoke(h)


# =========================================================================
# RFC 3161 Timestamp Authority
# =========================================================================

def obtain_rfc3161_timestamp(payload_digest:str,tsa_url:str)->bytes:
    """Obtain trusted timestamp for payload hash."""
    if not RFC3161_AVAILABLE:
        raise RuntimeError("RFC3161 client not installed")
    req=TimestampRequestBuilder().data(bytes.fromhex(payload_digest)).build()
    resp=requests.post(tsa_url,data=req.as_bytes(),
                       headers={"Content-Type":"application/timestamp-query"},
                       timeout=10,verify=certifi.where())
    resp.raise_for_status()
    return resp.content

def verify_rfc3161_timestamp(ts_bytes:bytes)->bool:
    """Validate timestamp structure (basic parse)."""
    if not RFC3161_AVAILABLE: return False
    try:
        decode_timestamp_response(ts_bytes)
        return True
    except Exception:
        return False


# =========================================================================
# Receipt Generation
# =========================================================================

def generate_receipt(
    code_str:str,inputs:Dict[str,Any],outputs:Dict[str,Any],
    executor_id:str,private_key_hex:str,*,
    version:str="4.1",meta:Optional[Dict[str,Any]]=None,
    nonce_registry:Optional[NonceRegistry]=None,
    revocation_registry:Optional[RevocationRegistry]=None,
    include_rfc3161:bool=False,tsa_url:Optional[str]=None,
    require_tsa:bool=False
)->Dict[str,Any]:
    """Generate a cryptographically verifiable receipt (fails-closed on collisions)."""
    canonical_code=_canonicalize_code(code_str)
    canonical_inputs=_canonical_json(inputs)
    canonical_outputs=_canonical_json(outputs)

    nonce=str(uuid.uuid4())
    if nonce_registry and not nonce_registry.register(nonce):
        raise ReceiptVerificationError("nonce","unique",nonce,"Duplicate nonce (replay)")

    payload={
        "version":version,
        "executor_id":executor_id,
        "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "nonce":nonce,
        "meta":meta or {},
        "code_hash":_sha256_hex(canonical_code),
        "input_hash":_sha256_hex(canonical_inputs),
        "output_hash":_sha256_hex(canonical_outputs)
    }

    sk=SigningKey(private_key_hex,encoder=HexEncoder)
    vk=sk.verify_key
    signature=sk.sign(_canonical_json(payload).encode()).signature

    receipt={
        "version":version,
        "payload":payload,
        "signature":signature.hex(),
        "public_key":vk.encode(encoder=HexEncoder).decode()
    }

    # Optional RFC3161 timestamp
    if include_rfc3161 and tsa_url:
        digest=_sha256_hex(_canonical_json(payload))
        try:
            ts_bytes=obtain_rfc3161_timestamp(digest,tsa_url)
            receipt["trusted_timestamp"]=ts_bytes.hex()
        except Exception as e:
            if require_tsa:
                raise ReceiptVerificationError("trusted_timestamp","present","missing",f"TSA failure: {e}")
            # else graceful degradation

    # Pre-generation revocation check
    if revocation_registry:
        payload_hash=_sha256_hex(_canonical_json(payload))
        if revocation_registry.is_revoked(payload_hash):
            raise ReceiptVerificationError("revocation","not revoked","revoked","Payload already revoked")

    return receipt


# =========================================================================
# Receipt Verification
# =========================================================================

def verify_receipt(
    receipt:Dict[str,Any],*,code_str=None,inputs=None,outputs=None,
    strict=False,nonce_registry=None,revocation_registry=None
)->bool:
    """Verify authenticity, revocation, replay protection, and optionally full execution context."""
    try:
        payload,signature_hex,pubkey_hex=receipt["payload"],receipt["signature"],receipt["public_key"]
    except KeyError as e:
        raise ReceiptVerificationError(str(e),None,None,"Missing field")

    vk=VerifyKey(pubkey_hex.encode(),encoder=HexEncoder)
    canonical=_canonical_json(payload).encode()
    try: vk.verify(canonical,bytes.fromhex(signature_hex))
    except BadSignatureError:
        raise ReceiptVerificationError("signature",None,signature_hex,"Invalid signature")

    if revocation_registry:
        payload_hash=_sha256_hex(_canonical_json(payload))
        if revocation_registry.is_revoked(payload_hash):
            raise ReceiptVerificationError("revoked","not revoked","revoked","Receipt revoked")

    if nonce_registry:
        nonce=payload["nonce"]
        if nonce_registry.seen(nonce):
            raise ReceiptVerificationError("nonce","<new>",nonce,"Replay detected")
        nonce_registry.register(nonce)

    if strict:
        if not all([code_str,inputs,outputs]):
            raise ReceiptVerificationError("context",None,None,"Strict mode requires code_str,inputs,outputs")
        expected={
            "code_hash":_sha256_hex(_canonicalize_code(code_str)),
            "input_hash":_sha256_hex(_canonical_json(inputs)),
            "output_hash":_sha256_hex(_canonical_json(outputs))
        }
        for f,exp in expected.items():
            if payload.get(f)!=exp:
                raise ReceiptVerificationError(f,exp,payload.get(f),f"Hash mismatch in {f}")

    if "trusted_timestamp" in receipt:
        ts_ok=verify_rfc3161_timestamp(bytes.fromhex(receipt["trusted_timestamp"]))
        if not ts_ok:
            raise ReceiptVerificationError("trusted_timestamp","valid","invalid","TSA timestamp invalid")

    return True


# =========================================================================
# Helpers: Batch Verification + Serialization
# =========================================================================

def verify_receipts_batch(receipts:List[Dict[str,Any]],**kwargs)->List[bool]:
    """Verify multiple receipts; return success flags."""
    return [verify_receipt(r,**kwargs) for r in receipts]

def receipt_to_json(receipt:Dict[str,Any])->str:
    return _canonical_json(receipt)

def receipt_from_json(js:str)->Dict[str,Any]:
    return json.loads(js)


# =========================================================================
# Key Utilities
# =========================================================================

def generate_keypair()->Dict[str,str]:
    """Generate Ed25519 keypair (Hex encoded)."""
    sk=SigningKey.generate()
    return {
        "private_key":sk.encode(encoder=HexEncoder).decode(),
        "public_key":sk.verify_key.encode(encoder=HexEncoder).decode()
    }


# =========================================================================
# Manual Test (Safe Demo)
# =========================================================================

if __name__=="__main__":
    path=Path("data"); path.mkdir(exist_ok=True)
    nonce_reg=NonceRegistry(path/"nonces.db")
    rev_reg=RevocationRegistry(path/"revocations.db")
    keys=generate_keypair()
    code="print('Hello Tessrax')"; inp, out={"x":2},{"result":4}

    r=generate_receipt(code,inp,out,"kernel",keys["private_key"],
                       nonce_registry=nonce_reg,revocation_registry=rev_reg,
                       include_rfc3161=False)
    print("Receipt:",json.dumps(r,indent=2))
    assert verify_receipt(r,code_str=code,inputs=inp,outputs=out,strict=True,
                          nonce_registry=nonce_reg,revocation_registry=rev_reg)
    print("✅ Verified")