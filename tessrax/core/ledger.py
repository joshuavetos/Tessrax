# tessrax/core/ledger.py
"""
Tessrax Immutable Ledger — A production-grade, append-only, tamper-evident event log.

Version: 2.1 (Lock Candidate)
Audit: A+ (Claude Audit 2025-10-04)
Status: ✅ Production-Ready

Metabolized Contradictions:
 • SCAR-001: Encoding mismatch → Standardized HexEncoder across stack.
 • SCAR-002: Fail-open revocation → Mandatory revocation enforcement.
 • SCAR-LEDGER-V2-001: Canonical storage pattern (tamper-evident reads).
 • SCAR-LEDGER-V2-002: Fail-closed posture (crash > corruption).

Features:
 • SQLite WAL mode for concurrent durability
 • Thread-safe serialized writes (Lock + BEGIN IMMEDIATE)
 • Canonical JSON storage (payload-only; hash == stored payload)
 • Signature + revocation verification before insertion
 • Persisted Merkle salt (derived from genesis)
 • Verified reads by default (detects tampering automatically)
 • Full chain verification (linkage + hash integrity)
 • Prometheus metrics for observability
 • Per-entry-type schema validation
 • Export/import utilities for audit and backup
 • Live statistics reporting
 • Documented serialization constraints (datetime, floats)
"""

import sqlite3
import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from nacl.signing import VerifyKey
from nacl.encoding import HexEncoder

# Internal imports
from tessrax.utils.metrics import EVENTS_COUNTER, MERKLE_LATENCY
from tessrax.core import receipts
from tessrax.core import revocation  # mandatory — fail closed if missing

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

DB_PATH = Path("data/ledger.db")
DB_PATH.parent.mkdir(exist_ok=True)

REQUIRED_FIELDS = {"entry_type", "timestamp"}

# JSON serialization guidelines:
# - Timestamps must be ISO 8601 strings.
# - Avoid raw datetime objects.
# - Use integers or strings for precise values (avoid floats for money).
#   Example: 12345 cents, not 123.45 dollars.

ENTRY_SCHEMAS: Dict[str, set] = {
    "receipt": {"entry_type", "timestamp", "receipt"},
    "audit": {"entry_type", "timestamp", "message"},
    "governance": {"entry_type", "timestamp", "decision", "votes"},
}

# -------------------------------------------------------------------------
# Ledger Implementation
# -------------------------------------------------------------------------

class SQLiteLedger:
    """
    Thread-safe, append-only, tamper-evident ledger.

    Stores canonical JSON payloads (without entry_hash) and the derived entry_hash.
    Each read re-verifies hash integrity; each write enforces schema + signature + revocation.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._initialize_db()
        self.lock = threading.Lock()
        self._ensure_merkle_salt()
        print("[Ledger] Initialized (WAL mode, thread-safe, revocation enforced).")

    # ---------------------------------------------------------------------
    # Schema Init
    # ---------------------------------------------------------------------

    def _initialize_db(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_json TEXT NOT NULL,
                    entry_hash TEXT NOT NULL UNIQUE,
                    prev_hash TEXT
                );
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

    # ---------------------------------------------------------------------
    # Canonicalization & Hashing
    # ---------------------------------------------------------------------

    @staticmethod
    def _canonical_json(obj: Dict[str, Any]) -> str:
        """Produce canonical JSON string representation (no whitespace, sorted keys)."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _hash_canonical_json(canonical: str) -> str:
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------------
    # Metadata / Merkle Salt
    # ---------------------------------------------------------------------

    def _get_metadata(self, key: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?;", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (key, value),
            )

    def _ensure_merkle_salt(self) -> None:
        """Persist a per-ledger Merkle salt derived from genesis entry."""
        if self._get_metadata("merkle_salt"):
            return
        cur = self.conn.cursor()
        cur.execute("SELECT entry_hash FROM events ORDER BY id ASC LIMIT 1;")
        row = cur.fetchone()
        seed = row[0] if row else "GENESIS"
        salt = hashlib.sha256(f"TESSRAX_ODD_NODE_SALT::{seed}".encode()).hexdigest()
        self._set_metadata("merkle_salt", salt)

    def _merkle_salt(self) -> str:
        salt = self._get_metadata("merkle_salt")
        if not salt:
            raise RuntimeError("Ledger missing Merkle salt — corruption detected.")
        return salt

    # ---------------------------------------------------------------------
    # Core Write Path
    # ---------------------------------------------------------------------

    def add_event(self, event: Dict[str, Any], verify: bool = True) -> str:
        """
        Adds an event after validation, schema enforcement, and signature checks.
        Returns computed entry_hash.
        """
        with self.lock:
            # Schema validation
            missing = REQUIRED_FIELDS - event.keys()
            if missing:
                raise ValueError(f"Event missing required fields: {missing}")

            entry_type = event.get("entry_type")
            if entry_type in ENTRY_SCHEMAS:
                required = ENTRY_SCHEMAS[entry_type]
                missing = required - event.keys()
                if missing:
                    raise ValueError(f"{entry_type} missing fields: {missing}")

            # Receipt verification
            if verify and "receipt" in event:
                receipt = event["receipt"]
                pubkey_hex = receipt.get("public_key") or receipt.get("signer_pubkey")
                if not pubkey_hex:
                    raise ValueError("Receipt missing public key.")
                try:
                    vk = VerifyKey(pubkey_hex.encode("ascii"), encoder=HexEncoder)
                except Exception as e:
                    raise ValueError(f"Invalid public key encoding: {e}")
                if not receipts.verify_receipt(vk, receipt):
                    raise ValueError("Invalid or tampered receipt signature.")
                if revocation.is_revoked(pubkey_hex):
                    raise ValueError(f"Signer {pubkey_hex} is revoked.")

            # Compute hash + linkage
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("BEGIN IMMEDIATE;")
                cursor.execute("SELECT entry_hash FROM events ORDER BY id DESC LIMIT 1;")
                last = cursor.fetchone()
                prev_hash = last[0] if last else None

                payload = event.copy()
                payload["prev_hash"] = prev_hash
                canonical_payload = self._canonical_json(payload)
                entry_hash = self._hash_canonical_json(canonical_payload)

                self.conn.execute(
                    "INSERT INTO events (event_json, entry_hash, prev_hash) VALUES (?, ?, ?);",
                    (canonical_payload, entry_hash, prev_hash),
                )
                EVENTS_COUNTER.inc()
                return entry_hash

    # ---------------------------------------------------------------------
    # Retrieval (Verified Reads)
    # ---------------------------------------------------------------------

    def get_all_events(self, verify: bool = True) -> List[Dict[str, Any]]:
        """Retrieve all events with optional verification (default: verify=True)."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT event_json, entry_hash FROM events ORDER BY id ASC;")
            results = []
            for canonical_json, stored_hash in cursor.fetchall():
                recalculated = self._hash_canonical_json(canonical_json)
                if verify and recalculated != stored_hash:
                    raise ValueError(
                        f"Tampered event detected: stored={stored_hash}, recalculated={recalculated}"
                    )
                event = json.loads(canonical_json)
                event["entry_hash"] = stored_hash
                results.append(event)
            return results

    # ---------------------------------------------------------------------
    # Merkle Root
    # ---------------------------------------------------------------------

    @MERKLE_LATENCY.time()
    def merkle_root(self) -> Optional[str]:
        """Compute salted Merkle root of all entry hashes."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT entry_hash FROM events ORDER BY id ASC;")
            hashes = [row[0] for row in cursor.fetchall()]

        if not hashes:
            return None

        salt = self._merkle_salt()
        level = hashes[:]
        while len(level) > 1:
            if len(level) % 2 != 0:
                level.append(salt)
            next_level = []
            for i in range(0, len(level), 2):
                combined = (level[i] + level[i + 1]).encode("utf-8")
                next_level.append(hashlib.sha256(combined).hexdigest())
            level = next_level
        return level[0]

    # ---------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify hash + linkage integrity for entire ledger."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT event_json, entry_hash, prev_hash FROM events ORDER BY id ASC;")
            rows = cursor.fetchall()

        if not rows:
            return True

        previous_hash = None
        for i, (canonical_json, stored_hash, prev_hash) in enumerate(rows):
            recalculated = self._hash_canonical_json(canonical_json)
            if stored_hash != recalculated:
                print(json.dumps({
                    "level": "error",
                    "event_index": i,
                    "type": "hash_mismatch",
                    "stored": stored_hash,
                    "recalculated": recalculated,
                }))
                return False
            if previous_hash and prev_hash != previous_hash:
                print(json.dumps({
                    "level": "error",
                    "event_index": i,
                    "type": "chain_break",
                    "prev_hash": prev_hash,
                    "expected": previous_hash,
                }))
                return False
            previous_hash = stored_hash
        return True

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def export_ledger(self, output_path: Path) -> None:
        """Export verified ledger to a JSON file."""
        events = self.get_all_events(verify=True)
        output_path.write_text(json.dumps(events, indent=2))
        print(f"[Ledger] Exported {len(events)} events to {output_path}")

    def import_ledger(self, input_path: Path) -> None:
        """Import ledger from JSON file (validates signatures)."""
        events = json.loads(input_path.read_text())
        for event in events:
            self.add_event(event, verify=True)
        print(f"[Ledger] Imported {len(events)} events from {input_path}")

    def stats(self) -> Dict[str, Any]:
        """Return ledger health metrics."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM events;")
            count, first, last = cursor.fetchone()
        return {
            "total_events": count,
            "first_timestamp": first,
            "last_timestamp": last,
            "merkle_root": self.merkle_root(),
            "chain_valid": self.verify_chain(),
        }

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------

    def close(self):
        self.conn.close()