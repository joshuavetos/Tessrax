"""Tier and usage tracking for the Truth API."""
from __future__ import annotations

from typing import Dict, Optional

from fastapi import HTTPException, status

from tessrax_truth_api.utils import load_config, utcnow


class BillingService:
    """Simple in-memory usage tracker with Stripe placeholders."""

    def __init__(self) -> None:
        config = load_config()
        self._tiers: Dict[str, Optional[int]] = {
            tier: details.get("limit") if isinstance(details, dict) else None
            for tier, details in config.get("billing", {}).get("tiers", {}).items()
        }
        self._usage: Dict[str, int] = {}
        self._last_reset = utcnow().date()

    def _reset_if_needed(self) -> None:
        today = utcnow().date()
        if today != self._last_reset:
            self._usage.clear()
            self._last_reset = today

    def record_usage(self, token: str, tier: str) -> None:
        """Record usage for the provided token."""

        self._reset_if_needed()
        limit = self._tiers.get(tier)
        if tier not in self._tiers:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported tier")
        current = self._usage.get(token, 0)
        if limit is not None and current >= limit:
            raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Tier usage exceeded")
        self._usage[token] = current + 1

    def usage_snapshot(self) -> Dict[str, int]:
        return dict(self._usage)
