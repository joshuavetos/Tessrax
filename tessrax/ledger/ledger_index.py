"""SQLite-backed index for fast historical claim lookup."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from tessrax.types import Claim


class LedgerIndex:
    """Maintain a SQLite index of claims emitted via the governance ledger."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.database_path = self._derive_database_path(self.ledger_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialise()

    @staticmethod
    def _derive_database_path(ledger_path: Path) -> Path:
        suffix = ledger_path.suffix or ""
        return (
            ledger_path.with_suffix(f"{suffix}.sqlite3")
            if suffix
            else ledger_path.with_name(f"{ledger_path.name}.sqlite3")
        )

    def _initialise(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT,
                subject TEXT NOT NULL,
                metric TEXT,
                value REAL,
                unit TEXT,
                source TEXT,
                timestamp TEXT,
                fingerprint TEXT UNIQUE
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_subject_metric ON claims(subject, metric)"
        )
        self._connection.commit()

    def sync(self, limit: int | None = None) -> None:
        """Load ledger entries from disk into the SQLite index."""

        if not self.ledger_path.exists():
            return

        try:
            entries = self._load_ledger_entries(limit)
        except OSError as exc:  # pragma: no cover - I/O issues
            logging.error("LedgerIndex sync failed to read ledger: %s", exc)
            return

        if not entries:
            return

        with self._lock:
            cursor = self._connection.cursor()
            for receipt in entries:
                contradiction = self._extract_contradiction(receipt)
                if not contradiction:
                    continue
                for key in ("claim_a", "claim_b"):
                    claim_data = contradiction.get(key)
                    if not isinstance(claim_data, dict):
                        continue
                    self._insert_claim(cursor, claim_data)
            self._connection.commit()

    def record_claim(self, claim: Claim) -> None:
        """Insert a single claim directly into the index."""

        payload = claim.to_json()
        self._insert_with_payload(payload)

    def query_similar(self, subject: str, metric: str | None = None) -> list[Claim]:
        """Return claims that share the subject (and optional metric)."""

        with self._lock:
            cursor = self._connection.cursor()
            if metric is None:
                cursor.execute(
                    "SELECT claim_id, subject, metric, value, unit, source, timestamp FROM claims WHERE subject = ?",
                    (subject,),
                )
            else:
                cursor.execute(
                    "SELECT claim_id, subject, metric, value, unit, source, timestamp FROM claims WHERE subject = ? AND metric = ?",
                    (subject, metric),
                )
            rows = cursor.fetchall()

        return [self._row_to_claim(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # Internal helpers -----------------------------------------------------

    def _insert_with_payload(self, payload: dict) -> None:
        with self._lock:
            cursor = self._connection.cursor()
            self._insert_claim(cursor, payload)
            self._connection.commit()

    def _insert_claim(self, cursor: sqlite3.Cursor, payload: dict) -> None:
        subject = payload.get("subject")
        metric = payload.get("metric")
        value = payload.get("value")
        source = payload.get("source")
        timestamp = payload.get("timestamp")
        if subject is None or value is None or source is None or timestamp is None:
            return

        unit_val = payload.get("unit", "dimensionless")
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        claim_id = payload.get("claim_id") or fingerprint[:8]

        cursor.execute(
            """
            INSERT OR IGNORE INTO claims(claim_id, subject, metric, value, unit, source, timestamp, fingerprint)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                claim_id,
                subject,
                metric,
                float(value),
                unit_val,
                source,
                str(timestamp),
                fingerprint,
            ),
        )

    def _load_ledger_entries(self, limit: int | None) -> Iterable[dict]:
        entries: Iterable[str]
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            if limit is None:
                entries = handle.readlines()
            else:
                buffer: deque[str] = deque(maxlen=limit)
                for line in handle:
                    buffer.append(line)
                entries = list(buffer)

        results = []
        for raw in entries:
            raw = raw.strip()
            if not raw:
                continue
            try:
                results.append(json.loads(raw))
            except json.JSONDecodeError:
                logging.warning("LedgerIndex ignoring invalid JSON line")
        return results

    @staticmethod
    def _extract_contradiction(receipt: dict) -> dict | None:
        decision = receipt.get("decision")
        if isinstance(decision, dict):
            contradiction = decision.get("contradiction")
            if isinstance(contradiction, dict):
                return contradiction
        contradiction = receipt.get("contradiction")
        if isinstance(contradiction, dict):
            return contradiction
        return None

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        timestamp_value = row["timestamp"]
        timestamp = LedgerIndex._parse_timestamp(timestamp_value)
        unit = row["unit"] or "dimensionless"
        metric = row["metric"] or ""
        return Claim(
            claim_id=row["claim_id"],
            subject=row["subject"],
            metric=metric,
            value=float(row["value"]),
            unit=unit,
            timestamp=timestamp,
            source=row["source"],
        )

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.fromtimestamp(float(value), tz=timezone.utc)
                except ValueError:
                    pass
        return datetime.now(timezone.utc)
