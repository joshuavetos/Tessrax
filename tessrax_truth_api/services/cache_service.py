"""Cache utilities for the Truth API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..utils import load_config


@dataclass
class CachedEntry:
    score: float
    verdict: str
    status: str


class CacheService:
    """A thin abstraction around Redis with an in-memory fallback."""

    def __init__(self) -> None:
        config = load_config()
        redis_config = config.get("redis", {})
        self._enabled = bool(redis_config.get("enabled", False))
        self._cache: Dict[Tuple[str, str], CachedEntry] = {}
        self._client = None
        if self._enabled:
            try:
                import redis

                self._client = redis.Redis.from_url(redis_config.get("url"))
            except Exception:  # pragma: no cover - optional dependency
                self._enabled = False

    def get(self, claim_a: str, claim_b: str) -> Optional[CachedEntry]:
        key = (claim_a.strip().lower(), claim_b.strip().lower())
        if self._enabled and self._client is not None:
            payload = self._client.get(str(key))
            if payload is None:
                return None
            score, verdict, status = payload.decode("utf-8").split("|")
            return CachedEntry(score=float(score), verdict=verdict, status=status)
        return self._cache.get(key)

    def set(self, claim_a: str, claim_b: str, entry: CachedEntry) -> None:
        key = (claim_a.strip().lower(), claim_b.strip().lower())
        if self._enabled and self._client is not None:
            payload = f"{entry.score}|{entry.verdict}|{entry.status}"
            self._client.set(str(key), payload, ex=3600)
        else:
            self._cache[key] = entry
