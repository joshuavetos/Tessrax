"""Claim extraction utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Mapping, Sequence

from .types import Claim


class ClaimExtractor:
    """Normalise heterogeneous records into structured :class:`Claim` objects."""

    def __init__(self, default_unit: str = "unit") -> None:
        self.default_unit = default_unit

    def extract(self, records: Sequence[Mapping[str, object]]) -> List[Claim]:
        """Convert iterable of mapping-like objects into claims.

        Each record must contain ``subject``, ``metric``, ``value``, and ``timestamp``.
        Optional fields: ``unit`` (defaults to ``self.default_unit``), ``source``, ``context``.
        """

        claims: List[Claim] = []
        for idx, record in enumerate(records):
            subject = str(record["subject"]).strip()
            metric = str(record["metric"]).strip()
            value = float(record["value"])
            unit = str(record.get("unit", self.default_unit))
            timestamp_raw = record.get("timestamp", datetime.now(timezone.utc).isoformat())
            timestamp = self._parse_timestamp(timestamp_raw)
            source = str(record.get("source", "unknown"))
            context = {
                key: str(val)
                for key, val in record.get("context", {}).items()
            }
            claim_id = record.get("claim_id") or f"claim-{idx+1}"
            claims.append(
                Claim(
                    claim_id=str(claim_id),
                    subject=subject,
                    metric=metric,
                    value=value,
                    unit=unit,
                    timestamp=timestamp,
                    source=source,
                    context=context,
                )
            )
        return claims

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")
