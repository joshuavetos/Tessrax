"""View-change management for the Tessrax federation consensus."""
from __future__ import annotations

import time
from typing import Iterable, List


class ViewManager:
    """Tracks consensus views and rotates leaders on timeout."""

    def __init__(self, node_ids: Iterable[str], view_timeout_ms: int = 150) -> None:
        self.node_ids: List[str] = list(node_ids)
        if not self.node_ids:
            raise ValueError("ViewManager requires at least one node identifier")
        self.view_timeout_ms = view_timeout_ms
        self.current_view = 0
        self._view_start = time.monotonic()

    def leader(self) -> str:
        index = self.current_view % len(self.node_ids)
        return self.node_ids[index]

    def maybe_advance(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        elapsed_ms = (now - self._view_start) * 1000.0
        if elapsed_ms >= self.view_timeout_ms:
            self.current_view += 1
            self._view_start = now
            return True
        return False

    def force_advance(self, new_view: int) -> None:
        if new_view < self.current_view:
            raise ValueError("Cannot regress views")
        self.current_view = new_view
        self._view_start = time.monotonic()


__all__ = ["ViewManager"]
