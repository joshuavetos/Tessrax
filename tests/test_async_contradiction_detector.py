"""Tests for the asynchronous contradiction detector."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

pytest.importorskip("numpy")

from tessrax.ledger import Ledger
from tessrax.metabolism.async_detector import AsyncContradictionDetector
from tessrax.types import Claim


@pytest.mark.asyncio
async def test_async_detection_emits_receipt() -> None:
    ledger = Ledger()
    detector = AsyncContradictionDetector(ledger, concurrency=2)

    now = datetime.now(timezone.utc)
    claim_a = Claim(
        claim_id="claim-a",
        subject="alpha",
        metric="temperature",
        value=10.0,
        unit="celsius",
        timestamp=now,
        source="sensor-a",
    )
    claim_b = Claim(
        claim_id="claim-b",
        subject="alpha",
        metric="temperature",
        value=15.5,
        unit="celsius",
        timestamp=now,
        source="sensor-b",
    )

    run_task = asyncio.create_task(detector.run())
    try:
        await detector.publish(claim_a)
        await detector.publish(claim_b)
        await asyncio.wait_for(detector.queue.join(), timeout=2)
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)
        await detector.shutdown()

    assert ledger.verify()
    receipts = ledger.receipts()
    assert receipts, "Expected at least one ledger receipt"
    assert receipts[0].decision.action == "ASYNC_DETECTION"
