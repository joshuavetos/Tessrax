"""Tests for the asynchronous contradiction detector."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

pytest.importorskip("numpy")

from tessrax.ledger import Ledger
from tessrax.metabolism.async_detector import AsyncContradictionDetector
from tessrax.types import Claim


def _make_claim(subject: str, metric: str, value: float, *, claim_id: str, source: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        subject=subject,
        metric=metric,
        value=value,
        unit="dimensionless",
        timestamp=datetime.now(timezone.utc),
        source=source,
    )


def test_backpressure_drops(tmp_path) -> None:
    async def scenario() -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        detector = AsyncContradictionDetector(ledger, maxsize=10)
        detector.workers = 1
        await detector.start()

        try:
            tasks = []
            for idx in range(50):
                claim = _make_claim(
                    subject="bp",
                    metric="m",
                    value=float(idx),
                    claim_id=f"bp-{idx}",
                    source="tester",
                )
                tasks.append(
                    asyncio.create_task(detector.publish(claim, timeout=0.001))
                )
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.1)
        finally:
            await detector.stop()

        metrics = detector.metrics()
        assert metrics["dropped"] > 0

    asyncio.run(scenario())


def test_deduplication(tmp_path) -> None:
    async def scenario() -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        detector = AsyncContradictionDetector(ledger)
        await detector.start()

        claim = _make_claim(
            subject="dup",
            metric="m",
            value=1.0,
            claim_id="dup-1",
            source="tester",
        )

        try:
            await detector.publish(claim)
            await detector.publish(claim)
            await asyncio.sleep(0.05)
        finally:
            await detector.stop()

        metrics = detector.metrics()
        assert metrics["claims_seen"] == 1

    asyncio.run(scenario())


def test_historical_contradiction(tmp_path) -> None:
    async def scenario() -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        detector = AsyncContradictionDetector(ledger)
        await detector.start()

        claim_a = _make_claim(
            subject="x",
            metric="temp",
            value=10.0,
            claim_id="claim-a",
            source="sensor-a",
        )
        await detector.publish(claim_a)
        await asyncio.sleep(0.1)
        history = detector.index.query_similar("x", "temp")
        assert len(history) == 1, "Expected claim_a to be indexed"

        claim_b = _make_claim(
            subject="x",
            metric="temp",
            value=-10.0,
            claim_id="claim-b",
            source="sensor-b",
        )
        await detector.publish(claim_b)
        await asyncio.sleep(0.2)

        await detector.stop()

        metrics = detector.metrics()
        assert metrics["detected"] >= 1
        receipts = ledger.receipts()
        assert receipts, "Expected a governance receipt"
        assert any(getattr(receipt.decision, "contradiction", None) is not None for receipt in receipts)

    asyncio.run(scenario())


def test_integration_metrics_and_shutdown(tmp_path) -> None:
    async def scenario() -> None:
        ledger = Ledger(tmp_path / "ledger.jsonl")
        detector = AsyncContradictionDetector(ledger)
        await detector.start()

        try:
            claim = _make_claim(
                subject="A",
                metric="temp",
                value=1.0,
                claim_id="metric-1",
                source="tester",
            )
            await detector.publish(claim)
            await asyncio.sleep(0.05)
        finally:
            await detector.stop()

        metrics = detector.metrics()
        assert metrics["published"] >= 1
        assert "queue_depth" in metrics
        assert "claims_seen" in metrics
        assert "workers" in metrics

    asyncio.run(scenario())
