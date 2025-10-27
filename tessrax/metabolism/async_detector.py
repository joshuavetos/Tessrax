"""Asynchronous contradiction detection pipeline with bounded queue and observability."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from tessrax.contradiction import ContradictionEngine
from tessrax.ledger import Ledger
from tessrax.ledger.ledger_index import LedgerIndex
from tessrax.types import Claim, ContradictionRecord, GovernanceDecision


class AsyncContradictionDetector:
    """Detect contradictions asynchronously with ledger-indexed history."""

    def __init__(self, ledger: Ledger, maxsize: int = 1000) -> None:
        self.ledger = ledger
        ledger_path = getattr(ledger, "path", None) or os.getenv("LEDGER_PATH", "./data/ledger.jsonl")
        self.index = LedgerIndex(ledger_path)

        try:
            queue_size = int(os.getenv("QUEUE_MAXSIZE", str(maxsize)))
        except ValueError:
            queue_size = maxsize
        self.queue: asyncio.Queue[Claim] = asyncio.Queue(maxsize=queue_size)
        self.engine = ContradictionEngine()
        self.workers = int(os.getenv("TESSRAX_ASYNC_WORKERS", "8"))

        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._sync_task: asyncio.Task[None] | None = None

        self._seen_tokens: set[str] = set()

        self.stats = dict(
            published=0,
            dropped=0,
            detected=0,
        )

    async def publish(self, claim: Claim, timeout: float = 1.0) -> None:
        """Attempt to enqueue a claim for processing."""

        try:
            await asyncio.wait_for(self.queue.put(claim), timeout)
            self.stats["published"] += 1
        except asyncio.TimeoutError:
            self.stats["dropped"] += 1
            logging.warning("AsyncContradictionDetector backpressure: claim dropped")

    async def worker(self) -> None:
        """Process claims from the queue."""

        while True:
            try:
                claim = await self.queue.get()
            except asyncio.CancelledError:
                break

            token = self._claim_token(claim)
            if token in self._seen_tokens:
                self.queue.task_done()
                continue
            self._seen_tokens.add(token)

            try:
                historical = self.index.query_similar(claim.subject, claim.metric)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("LedgerIndex query failed: %s", exc)
                historical = []

            try:
                contradictions = await self.detect_async(claim, historical)
                await self._record_results(claim, contradictions)
            finally:
                self.queue.task_done()

    async def detect_async(self, claim: Claim, historical: Iterable[Claim]) -> List[ContradictionRecord]:
        """Run contradiction detection in a background executor."""

        loop = asyncio.get_running_loop()
        claims = [claim, *historical]
        return await loop.run_in_executor(None, self.engine.detect, claims)

    async def _record_results(self, claim: Claim, contradictions: Iterable[ContradictionRecord]) -> None:
        """Append governance decisions and refresh the ledger index."""

        decisions = list(contradictions)
        for record in decisions:
            decision = self._to_ledger_record(record)
            self.ledger.append(decision)
            self.stats["detected"] += 1
            for claim in (record.claim_a, record.claim_b):
                try:
                    self.index.record_claim(claim)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logging.error("LedgerIndex record_claim failed: %s", exc)

        # Ensure the triggering claim is available for future comparisons.
        try:
            self.index.record_claim(claim)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("LedgerIndex record_claim failed: %s", exc)

    def _claim_token(self, claim: Claim) -> str:
        timestamp = claim.timestamp.isoformat()
        return f"{claim.key()}::{claim.claim_id}::{timestamp}::{claim.value}"

    def _to_ledger_record(self, contradiction: ContradictionRecord) -> GovernanceDecision:
        rationale = (
            f"Detected asynchronous contradiction between {contradiction.claim_a.claim_id} "
            f"and {contradiction.claim_b.claim_id}: {contradiction.reasoning}"
        )
        return GovernanceDecision(
            contradiction=contradiction,
            action="ASYNC_DETECTION",
            clarity_fuel=max(0.1, contradiction.delta),
            rationale=rationale,
            protocol="ASYNC_METABOLISM_V1",
        )

    async def start(self) -> None:
        """Initialise worker pool and background sync."""

        if self._running:
            return
        self._running = True
        self._seen_tokens.clear()
        self.index.sync(limit=None)
        self._tasks = [asyncio.create_task(self.worker()) for _ in range(self.workers)]
        self._sync_task = asyncio.create_task(self._periodic_sync())

    async def stop(self) -> None:
        """Gracefully stop workers and close the ledger index."""

        if not self._running:
            return

        final_depth = self.queue.qsize()
        if final_depth > 0:
            logging.warning(
                "AsyncContradictionDetector shutdown with %s unprocessed claims", final_depth
            )

        self._running = False
        await self.queue.join()

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._sync_task is not None:
            self._sync_task.cancel()
            await asyncio.gather(self._sync_task, return_exceptions=True)
            self._sync_task = None

        self.index.close()

    async def _periodic_sync(self, interval: int = 60) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            try:
                self.index.sync()
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("LedgerIndex periodic sync error: %s", exc)

    def metrics(self) -> dict[str, int]:
        return {
            "queue_depth": self.queue.qsize(),
            "claims_seen": len(self._seen_tokens),
            "workers": self.workers,
            **self.stats,
        }

    @property
    def seen(self) -> set[str]:
        """Expose the deduplicated claim tokens for compatibility."""

        return set(self._seen_tokens)


async def _run_self_test() -> None:
    logging.basicConfig(level=logging.INFO)
    tmp_path = Path("./out/async_detector_self_test")
    tmp_path.mkdir(parents=True, exist_ok=True)

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    detector = AsyncContradictionDetector(ledger, maxsize=10)
    await detector.start()

    try:
        now = datetime.now(timezone.utc)
        claim_a = Claim(
            claim_id="self-test-a",
            subject="async",
            metric="temperature",
            value=10.0,
            unit="celsius",
            timestamp=now,
            source="sensor-a",
        )
        claim_b = Claim(
            claim_id="self-test-b",
            subject="async",
            metric="temperature",
            value=-10.0,
            unit="celsius",
            timestamp=now,
            source="sensor-b",
        )

        await detector.publish(claim_a)
        await detector.publish(claim_b)
        await asyncio.sleep(0.5)
    finally:
        await detector.stop()

    logging.info("AsyncContradictionDetector metrics: %s", detector.metrics())


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tessrax Async Contradiction Detector utilities")
    parser.add_argument("--self-test", action="store_true", help="Run the async detector self-test")
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    if args.self_test:
        asyncio.run(_run_self_test())
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
