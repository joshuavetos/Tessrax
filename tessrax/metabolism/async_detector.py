"""Asynchronous contradiction detection pipeline."""

from __future__ import annotations

import asyncio
from typing import List

from tessrax.contradiction import ContradictionEngine
from tessrax.ledger import Ledger
from tessrax.types import Claim, ContradictionRecord, GovernanceDecision


class AsyncContradictionDetector:
    """Run contradiction detection in the background using :mod:`asyncio`."""

    def __init__(self, ledger: Ledger, concurrency: int = 8) -> None:
        self.ledger = ledger
        self.concurrency = max(1, concurrency)
        self.queue: asyncio.Queue[Claim] = asyncio.Queue()
        self.engine = ContradictionEngine()
        self.seen: set[str] = set()
        self.claim_store: List[Claim] = []
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []

    async def publish(self, claim: Claim) -> None:
        """Submit a claim for asynchronous contradiction evaluation."""

        await self.queue.put(claim)

    async def worker(self) -> None:
        """Continuously process queued claims."""

        while True:
            try:
                claim = await self.queue.get()
            except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
                break

            claim_key = claim.key()
            seen_token = f"{claim_key}::{claim.claim_id}"
            if seen_token in self.seen:
                self.queue.task_done()
                continue

            self.seen.add(seen_token)

            try:
                contradictions = await self.detect_async(claim)
                for contradiction in contradictions:
                    decision = self._to_ledger_record(contradiction)
                    self.ledger.append(decision)
            finally:
                self.queue.task_done()

    async def detect_async(self, claim: Claim) -> List[ContradictionRecord]:
        """Run contradiction detection for a claim in a background executor."""

        async with self._lock:
            related_claims = [
                stored_claim for stored_claim in self.claim_store if stored_claim.key() == claim.key()
            ]
            related_claims.append(claim)
            self.claim_store.append(claim)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.engine.detect, related_claims)
        claim_id = claim.claim_id
        return [
            record
            for record in results
            if claim_id in (record.claim_a.claim_id, record.claim_b.claim_id)
        ]

    def _to_ledger_record(self, contradiction: ContradictionRecord) -> GovernanceDecision:
        """Convert a contradiction into a ledger governance decision."""

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

    async def run(self) -> None:
        """Spawn worker tasks and keep them alive until cancelled."""

        self.start()
        try:
            await self.queue.join()
        finally:
            await self._cancel_workers()

    def start(self) -> None:
        """Ensure worker tasks are running."""

        if self._workers:
            return
        self._workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]

    async def shutdown(self) -> None:
        """Cancel worker tasks and wait for them to finish."""

        await self.queue.join()
        await self._cancel_workers()

    async def _cancel_workers(self) -> None:
        if not self._workers:
            return
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
