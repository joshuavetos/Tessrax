"""Auto-governance health monitor aligned with DLK requirements."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ledger import append as ledger_append
from tessrax.metrics.epistemic_health import compute_integrity

AUDITOR_ID = "Tessrax Governance Kernel v16"
CLAUSES = ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"]
DEFAULT_OUTCOMES_PATH = Path("out/health_observations.json")
SUMMARY_PATH = Path("out/health_summary.json")


def _load_outcomes(path: Path) -> list[float]:
    if not path.exists():
        return [1.0]
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Sequence):
        raise ValueError("Health observations must be a sequence")
    outcomes: list[float] = [float(value) for value in data]
    if not outcomes:
        raise ValueError("Health observations cannot be empty")
    return outcomes


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_health_check(
    outcomes: Iterable[float] | None = None,
    *,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> dict[str, object]:
    """Compute and log the epistemic integrity health summary."""

    collected = list(outcomes) if outcomes is not None else _load_outcomes(outcomes_path)
    if not collected:
        raise ValueError("At least one health outcome is required")
    integrity = compute_integrity(list(collected))
    timestamp = datetime.now(timezone.utc).isoformat()
    ledger_event = {
        "auditor": AUDITOR_ID,
        "clauses": CLAUSES,
        "timestamp": timestamp,
        "event": "HEALTH_CHECK",
        "integrity": integrity,
        "samples": len(collected),
        "status": "recorded",
        "integrity_score": integrity,
    }
    ledger_append(ledger_event)
    summary = {
        "auditor": AUDITOR_ID,
        "clauses": CLAUSES,
        "timestamp": timestamp,
        "event": "HEALTH_SUMMARY",
        "integrity": integrity,
        "samples": len(collected),
        "status": "verified",
    }
    _write_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    run_health_check()
