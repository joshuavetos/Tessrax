"""
tessrax/core/ledger.py
-----------------------
SQLiteLedger — Append-only, tamper-evident ledger implementing ILedger.

Core Guarantees:
✓ Tamper-evident (Merkle-linked)
✓ Thread-safe with atomic inserts
✓ Compatible with ILedger contract
✓ Auditable: full verify_chain() and merkle_root()
"""

import sqlite3
import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from tessrax.core.interfaces import ILedger
from prometheus_client import Counter, Histogram


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
LEDGER_WRITES = Counter("tessrax_ledger_writes_total", "Number of events appended to the ledger")
LEDGER_VERIFY_LATENCY = Histogram("tessrax_ledger_verify_seconds", "Time spent verifying ledger integrity")


# ------------------------------------------------------------
# SQLiteLedger Implementation
# ------------------------------------------------------------
class SQLiteLedger(ILedger):
    """
    Thread-safe, append-only, tamper-evident ledger backed by SQLite.
    Implements the ILedger API contract.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._init_schema()

    # --------------------------------------------------------
    # Schema Initialization
    # --------------------------------------------------------
    def _init_schema(self):
        """Create table if missing."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    entry_hash TEXT UNIQUE NOT NULL,
                    prev_hash TEXT,
                    payload TEXT NOT NULL
                )
            """)

    # --------------------------------------------------------
    # Core Interface Methods
    # --------------------------------------------------------
    def add_event(self, event: Dict[str, Any]) -> str:
        """
        Adds an event to the ledger.
        Computes its entry_hash linked to the previous entry.
        """
        with self._lock:
            last_hash = self._get_last_hash()
            payload_str = json.dumps(event, sort_keys=True)
            combined = (payload_str + (last_hash or "")).encode()
            entry_hash = hashlib.sha256(combined).hexdigest()

            record = {
                "timestamp": event.get("timestamp"),
                "entry_hash": entry_hash,
                "prev_hash": last_hash,
                "payload": payload_str,
            }

            with self.conn:
                self.conn.execute(
                    "INSERT INTO ledger (timestamp, entry_hash, prev_hash, payload) VALUES (?, ?, ?, ?)",
                    (record["timestamp"], record["entry_hash"], record["prev_hash"], record["payload"]),
                )
            LEDGER_WRITES.inc()
            return entry_hash

    def get_all_events(self, verify: bool = True) -> List[Dict[str, Any]]:
        """Retrieve all ledger events in order."""
        cursor = self.conn.execute("SELECT payload FROM ledger ORDER BY id ASC")
        rows = cursor.fetchall()
        events = [json.loads(r[0]) for r in rows]
        if verify:
            self.verify_chain()
        return events

    @LEDGER_VERIFY_LATENCY.time()
    def verify_chain(self) -> bool:
        """Verify hash linkage for all entries."""
        cursor = self.conn.execute("SELECT entry_hash, prev_hash, payload FROM ledger ORDER BY id ASC")
        rows = cursor.fetchall()
        last_hash = None
        for i, (entry_hash, prev_hash, payload) in enumerate(rows):
            expected_hash = hashlib.sha256((payload + (last_hash or "")).encode()).hexdigest()
            if expected_hash != entry_hash or prev_hash != last_hash:
                raise ValueError(f"Ledger corruption detected at index {i}")
            last_hash = entry_hash
        return True

    def merkle_root(self) -> Optional[str]:
        """Compute Merkle root of all entries."""
        cursor = self.conn.execute("SELECT entry_hash FROM ledger ORDER BY id ASC")
        hashes = [r[0] for r in cursor.fetchall()]
        if not hashes:
            return None
        while len(hashes) > 1:
            it = iter(hashes)
            new_level = [
                hashlib.sha256((a + (next(it, a))).encode()).hexdigest()
                for a in it
            ]
            hashes = new_level
        return hashes[0]

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _get_last_hash(self) -> Optional[str]:
        cursor = self.conn.execute("SELECT entry_hash FROM ledger ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        """Close database connection."""
        self.conn.close()