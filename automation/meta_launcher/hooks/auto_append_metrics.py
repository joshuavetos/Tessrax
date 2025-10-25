"""
Auto-append latest epistemic metrics to ledger.log after successful test run.
Triggered post-pytest via CI workflow.
"""

import json
import datetime
import subprocess
from pathlib import Path

from tessrax.metrics.epistemic_health import (
    compute_integrity,
    compute_drift,
    compute_severity,
    compute_entropy,
)


def capture_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def append_metrics_to_ledger(outcomes, expected, observed, labels) -> None:
    integrity = compute_integrity(outcomes)
    drift = compute_drift([(0, integrity)])  # simplified placeholder
    severity = compute_severity(expected, observed)
    entropy = compute_entropy(labels)
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "commit": capture_commit_hash(),
        "directive": "AUTO_METRIC_APPEND",
        "integrity": round(integrity, 3),
        "drift": round(drift, 3),
        "severity": round(severity, 3),
        "entropy": round(entropy, 3),
        "status": "success",
    }
    path = Path("automation/meta_launcher/logs/ledger.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    sample_outcomes = [0.91, 0.93, 0.89]
    sample_expected = [0.9, 0.85, 0.8]
    sample_observed = [0.92, 0.84, 0.79]
    sample_labels = ["semantic", "procedural", "semantic"]
    append_metrics_to_ledger(sample_outcomes, sample_expected, sample_observed, sample_labels)
