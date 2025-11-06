"""Governance kernel extensions with field evidence integration and repair loops."""

from __future__ import annotations

import asyncio
import signal
from contextlib import suppress
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from tessrax.data.evidence_loader import load_field_evidence
from tessrax.governance import GovernanceKernel as BaseGovernanceKernel
from tessrax.core.integrity_monitor import IntegrityMonitor
from tessrax.core.key_vault import KeyVault
from tessrax.core.ledger import Ledger
from tessrax.core.repair_engine import RepairEngine


class GovernanceKernel(BaseGovernanceKernel):
    """Governance kernel that bootstraps field evidence archives."""

    def __init__(self, *args: Any, auto_integrate: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._field_evidence: list[dict[str, Any]] = []
        self._field_evidence_summary: dict[str, Any] = {}
        self._ledger_path = Path(".ledger.jsonl")
        self._ledger = Ledger(self._ledger_path)
        self.key_vault = KeyVault(self._ledger, path=Path("out/keys"))
        self.integrity_monitor = IntegrityMonitor(ledger_path=self._ledger_path)
        self.repair_engine = RepairEngine(self._ledger)
        self._background_tasks: list[asyncio.Task[Any]] = []
        if auto_integrate:
            self.integrate_field_evidence()

    @property
    def field_evidence(self) -> list[dict[str, Any]]:
        """Return the cached field evidence entries."""

        return [entry.copy() for entry in self._field_evidence]

    def integrate_field_evidence(self, *, refresh: bool = False) -> dict[str, Any]:
        """Load field evidence and compute contradiction/policy alignment indicators."""

        if not self._field_evidence or refresh:
            self._field_evidence = load_field_evidence()
            self._field_evidence_summary = self._summarise_field_evidence(
                self._field_evidence
            )
        return dict(self._field_evidence_summary)

    async def start_background_services(self) -> None:
        """Launch integrity monitoring and repair loops."""

        loop = asyncio.get_running_loop()
        if any(not task.done() for task in self._background_tasks):
            return
        monitor_task = loop.create_task(self.integrity_monitor.monitor(), name="integrity-monitor")
        repair_task = loop.create_task(self.repair_engine.monitor(), name="repair-engine")
        self._background_tasks = [monitor_task, repair_task]
        self._register_signal_handlers(loop)

    async def shutdown_background_services(self) -> None:
        """Cancel background monitoring tasks gracefully."""

        if not self._background_tasks:
            return
        for task in self._background_tasks:
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

    def _register_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(self.shutdown_background_services()))
            except (NotImplementedError, RuntimeError):  # pragma: no cover - platform dependent
                pass

    @property
    def background_tasks(self) -> list[asyncio.Task[Any]]:
        """Expose background tasks for observability and testing."""

        return list(self._background_tasks)

    def _summarise_field_evidence(
        self, records: Iterable[Mapping[str, Any]]
    ) -> dict[str, Any]:
        category_counts: dict[str, int] = defaultdict(int)
        alignment_scores: dict[str, list[float]] = defaultdict(list)
        contradiction_cases: list[str] = []

        for record in records:
            category = str(record.get("category", "uncategorised"))
            category_counts[category] += 1

            alignment = record.get("alignment")
            if isinstance(alignment, Mapping):
                policy = str(alignment.get("policy_reference", "unspecified"))
                score = alignment.get("score")
                if isinstance(score, (int, float)):
                    alignment_scores[policy].append(float(score))

            searchable_text = " ".join(
                [
                    str(record.get("summary", "")),
                    " ".join(record.get("key_findings", [])),
                ]
            ).lower()
            if "contradiction" in searchable_text:
                record_id = str(record.get("id", "")) or category
                contradiction_cases.append(record_id)

        averaged_alignment = {
            policy: sum(scores) / len(scores)
            for policy, scores in alignment_scores.items()
            if scores
        }

        return {
            "total_records": sum(category_counts.values()),
            "category_counts": dict(category_counts),
            "alignment_scores": averaged_alignment,
            "contradiction_cases": contradiction_cases,
        }


__all__ = ["GovernanceKernel"]
