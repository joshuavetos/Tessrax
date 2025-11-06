"""Tests for the continuous integrity monitor."""

from __future__ import annotations

import asyncio
import tempfile

from tessrax.core.integrity_monitor import IntegrityMonitor


def test_monitor_cycles_once() -> None:
    path = tempfile.mktemp()
    monitor = IntegrityMonitor(path)

    async def _runner() -> None:
        await asyncio.wait_for(monitor.monitor(interval=0.1), timeout=0.3)

    try:
        asyncio.run(_runner())
    except asyncio.TimeoutError:
        pass

    assert len(monitor.ledger.receipts()) >= 1
