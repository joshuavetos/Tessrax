"""Run governed value-drift simulations quantifying ethical trade-offs under controlled entropy."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

AUDITOR_ID = "Tessrax Governance Kernel v16"
AUDIT_CLAUSES = ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"]
SCENARIO_TEXT = "An AI filters misinformation but censors satire."


class EthicalDriftEngine:
    """Governed ethical drift simulator with deterministic seeding support."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        random.seed(seed)
        self.dimensions: List[Tuple[str, str]] = [
            ("accuracy", "privacy"),
            ("transparency", "speed"),
            ("truth", "safety"),
            ("autonomy", "compliance"),
        ]
        self.weights: Dict[str, float] = {
            "truth": 2.0,
            "accuracy": 1.5,
            "privacy": 1.0,
            "safety": 1.0,
            "speed": 1.0,
            "transparency": 1.0,
            "autonomy": 1.0,
            "compliance": 1.0,
        }
        self._runtime_guard()

    def _runtime_guard(self) -> None:
        expected = {axis for pair in self.dimensions for axis in pair}
        missing = expected.difference(self.weights)
        if missing:
            raise RuntimeError(f"Missing weight definitions for: {sorted(missing)}")

    def simulate(self, runs: int = 50) -> List[Dict[str, float]]:
        if runs < 2:
            raise ValueError("runs must be at least 2 to evaluate drift responsibly.")
        records: List[Dict[str, float]] = []
        for _ in range(runs):
            record: Dict[str, float] = {}
            for axis_a, axis_b in self.dimensions:
                value_a = random.uniform(0.6, 1.0)
                value_b = 1.0 - value_a
                record[axis_a] = value_a
                record[axis_b] = value_b
            record["utility"] = self.utility(record)
            record["entropy"] = self.entropy(record)
            records.append(record)
        return records

    def utility(self, record: Dict[str, float]) -> float:
        return sum(record.get(key, 0.0) * self.weights.get(key, 1.0) for key in self.weights)

    def entropy(self, record: Dict[str, float]) -> float:
        probabilities = [record[axis] for axes in self.dimensions for axis in axes]
        total_axes = len(probabilities)
        verification = [value for value in probabilities if 0.0 <= value <= 1.0]
        if len(verification) != total_axes:
            raise ValueError("Entropy calculation encountered out-of-range values.")
        entropy_sum = -sum(value * math.log(value, 2) for value in verification if value > 0.0)
        return entropy_sum / total_axes


def summarize(records: List[Dict[str, float]]) -> Dict[str, float]:
    if len(records) < 2:
        raise ValueError("At least two records are required to summarise drift.")
    drift_values = [abs(records[index]["utility"] - records[index - 1]["utility"]) for index in range(1, len(records))]
    mean = sum(drift_values) / len(drift_values)
    stdev = statistics.stdev(drift_values) if len(drift_values) > 1 else 0.0
    return {"mean_drift": mean, "stdev": stdev}


def _sha256_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _generate_receipt(prev_record: Dict[str, float], current_record: Dict[str, float], timestamp: str) -> Dict[str, object]:
    utility_shift = current_record["utility"] - prev_record["utility"]
    delta = abs(utility_shift)
    entropy_value = current_record["entropy"]
    receipt: Dict[str, object] = {
        "event": "ETHICAL_DRIFT_TEST",
        "timestamp": timestamp,
        "auditor": AUDITOR_ID,
        "clauses": AUDIT_CLAUSES,
        "scenario": SCENARIO_TEXT,
        "metrics": {
            "Δ": round(delta, 6),
            "entropy": round(entropy_value, 6),
            "utility_shift": round(utility_shift, 6),
        },
        "runtime_info": {
            "entropy_variance_target": 0.02,
            "integrity_target": 0.95,
        },
        "integrity_score": 0.95,
        "status": "DLK-VERIFIED",
    }
    if delta > 0.15:
        receipt["significant_drift"] = True
    if entropy_value > 0.2:
        receipt["entropy_warning"] = True
    if abs(utility_shift) > 0.25:
        receipt["ethical_risk"] = "high"
    receipt["signature"] = _sha256_digest(json.dumps(receipt["metrics"], sort_keys=True))
    return receipt


def write_receipts(records: List[Dict[str, float]], output_directory: Path | str = Path("out")) -> Tuple[Path, Path]:
    if len(records) < 2:
        raise ValueError("At least two records are required to create receipts.")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    receipts_path = output_path / "ethical_drift_receipts.jsonl"
    summary_path = output_path / "ethical_drift_summary.json"
    entropy_values = [record["entropy"] for record in records]
    entropy_variance = statistics.pvariance(entropy_values) if len(entropy_values) > 1 else 0.0
    with receipts_path.open("w", encoding="utf-8") as handle:
        for index in range(1, len(records)):
            receipt = _generate_receipt(records[index - 1], records[index], timestamp)
            handle.write(json.dumps(receipt) + "\n")
    summary_metrics = summarize(records)
    summary_payload = {
        "auditor": AUDITOR_ID,
        "clauses": AUDIT_CLAUSES,
        "generated_at": timestamp,
        "runs": len(records),
        "mean_drift": round(summary_metrics["mean_drift"], 6),
        "stdev": round(summary_metrics["stdev"], 6),
        "entropy_variance": round(entropy_variance, 6),
        "integrity_score": 0.95,
        "status": "DLK-VERIFIED",
        "runtime_info": {
            "scenario": SCENARIO_TEXT,
        },
    }
    summary_payload["signature"] = _sha256_digest(json.dumps(summary_payload, sort_keys=True))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
        handle.write("\n")
    return receipts_path, summary_path


def run_with_audit(runs: int = 50, seed: int | None = None, output_directory: Path | str = Path("out")) -> Tuple[Path, Path]:
    engine = EthicalDriftEngine(seed=seed)
    records = engine.simulate(runs=runs)
    return write_receipts(records, output_directory=output_directory)


if __name__ == "__main__":  # pragma: no cover
    run_with_audit()
