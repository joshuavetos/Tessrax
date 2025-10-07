# tessrax/core/ledger.py
# Rewritten SQLiteLedger for Tessrax — v2.0
# Author: Joshua Vetos / Rewritten by OpenAI GPT-4o
# License: CC BY 4.0

import sqlite3
import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from prometheus_client import Counter, Histogram

from tessrax.core.interfaces import ILedger

# ----------------------------------------
# Prometheus Metrics
# ----------------------------------------
LEDGER_WRITES = Counter("tessrax_ledger_writes_total", "Number of events appended to the ledger")
LEDGER_VERIFY_LATENCY = Histogram("tessrax_ledger_verify_seconds", "Time spent verifying ledger integrity")


# ----------------------------------------
# Data Structure
# ----------------------------------------
@dataclass
class LedgerEntry:
    timestamp: str
    entry_hash: str
    prev_hash: Optional[str]
    payload: str  # JSON string

    def to_tuple(self):
        return (self.timestamp, self.entry_hash, self.prev_hash, self.payload)


# ----------------------------------------
# SQLiteLedger Implementation
# ----------------------------------------
class SQLiteLedger(ILedger):
    """
    Append-only, Merkle-linked, thread-safe ledger backed by SQLite.
    Implements ILedger contract for Tessrax systems.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        """Ensure the ledger table exists."""
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

    # ----------------------------------------
    # ILedger Methods
    # ----------------------------------------

    def add_event(self, event: Dict[str, Any]) -> str:
        """
        Add a new event to the ledger.
        Computes linked entry_hash using the payload + previous hash.
        """
        with self._lock:
            prev_hash = self._get_last_hash()
            payload_str = json.dumps(event, sort_keys=True)
            entry_hash = self._compute_hash(payload_str, prev_hash)

            entry = LedgerEntry(
                timestamp=event.get("timestamp"),
                entry_hash=entry_hash,
                prev_hash=prev_hash,
                payload=payload_str
            )

            with self.conn:
                self.conn.execute(
                    "INSERT INTO ledger (timestamp, entry_hash, prev_hash, payload) VALUES (?, ?, ?, ?)",
                    entry.to_tuple()
                )

            LEDGER_WRITES.inc()
            return entry_hash

    def get_all_events(self, verify: bool = True) -> List[Dict[str, Any]]:
        """Return all events from the ledger, verifying chain if requested."""
        rows = self.conn.execute("SELECT payload FROM ledger ORDER BY id ASC").fetchall()
        events = [json.loads(r[0]) for r in rows]

        if verify:
            self.verify_chain()

        return events

    @LEDGER_VERIFY_LATENCY.time()
    def verify_chain(self) -> bool:
        """Ensure every entry's hash chain is intact."""
        rows = self.conn.execute(
            "SELECT entry_hash, prev_hash, payload FROM ledger ORDER BY id ASC"
        ).fetchall()

        last_hash = None
        for i, (entry_hash, prev_hash, payload) in enumerate(rows):
            expected = self._compute_hash(payload, last_hash)

            if entry_hash != expected:
                raise ValueError(f"[Ledger Verify] Invalid hash at index {i}: expected {expected}, got {entry_hash}")
            if prev_hash != last_hash:
                raise ValueError(f"[Ledger Verify] Invalid chain link at index {i}: expected {last_hash}, got {prev_hash}")

            last_hash = entry_hash

        return True

    def merkle_root(self) -> Optional[str]:
        """Compute the Merkle root from all entry hashes."""
        hashes = [
            row[0]
            for row in self.conn.execute("SELECT entry_hash FROM ledger ORDER BY id ASC").fetchall()
        ]

        if not hashes:
            return None

        return self._compute_merkle_root(hashes)

    def close(self):
        """Close the database connection."""
        self.conn.close()

    # ----------------------------------------
    # Internal Utilities
    # ----------------------------------------

    def _get_last_hash(self) -> Optional[str]:
        row = self.conn.execute("SELECT entry_hash FROM ledger ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def _compute_hash(self, payload: str, prev_hash: Optional[str]) -> str:
        combined = (payload + (prev_hash or "")).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def _compute_merkle_root(self, hashes: List[str]) -> str:
        """
        Compute Merkle root from a list of hashes.
        Handles uneven trees by duplicating last hash in each round.
        """
        current = hashes[:]
        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])  # duplicate last
            current = [
                hashlib.sha256((current[i] + current[i + 1]).encode()).hexdigest()
                for i in range(0, len(current), 2)
            ]
        return current[0]