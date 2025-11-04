from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessrax.ethics.ethical_drift_engine import (
    AUDIT_CLAUSES,
    AUDITOR_ID,
    SCENARIO_TEXT,
    EthicalDriftEngine,
    summarize,
    write_receipts,
)


@pytest.fixture()
def seeded_records() -> list[dict[str, float]]:
    engine = EthicalDriftEngine(seed=42)
    return engine.simulate(runs=5)


def test_simulation_is_deterministic(seeded_records: list[dict[str, float]]) -> None:
    engine = EthicalDriftEngine(seed=42)
    repeated = engine.simulate(runs=5)
    for original, repeat in zip(seeded_records, repeated, strict=True):
        for key in ("accuracy", "privacy", "transparency", "speed", "truth", "safety", "autonomy", "compliance"):
            assert repeat[key] == pytest.approx(original[key], rel=1e-12)
    assert repeated[0]["utility"] == pytest.approx(5.137897087039224, rel=1e-12)
    assert repeated[0]["entropy"] == pytest.approx(0.41533867501888133, rel=1e-12)


def test_summary_and_receipts(tmp_path: Path, seeded_records: list[dict[str, float]]) -> None:
    metrics = summarize(seeded_records)
    assert metrics["mean_drift"] == pytest.approx(0.19166128245074732, rel=1e-12)
    assert metrics["stdev"] == pytest.approx(0.1385803417258314, rel=1e-12)

    receipts_path, summary_path = write_receipts(seeded_records, output_directory=tmp_path)
    assert receipts_path.exists()
    assert summary_path.exists()

    first_line = receipts_path.read_text(encoding="utf-8").splitlines()[0]
    receipt = json.loads(first_line)
    assert receipt["event"] == "ETHICAL_DRIFT_TEST"
    assert receipt["scenario"] == SCENARIO_TEXT
    assert receipt["auditor"] == AUDITOR_ID
    assert receipt["clauses"] == AUDIT_CLAUSES
    assert receipt["metrics"]["Δ"] == pytest.approx(0.26626898287551626, rel=1e-12)

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_data["auditor"] == AUDITOR_ID
    assert summary_data["clauses"] == AUDIT_CLAUSES
    assert summary_data["mean_drift"] == pytest.approx(0.191661, rel=1e-6)
    assert summary_data["stdev"] == pytest.approx(0.13858, rel=1e-5)
    assert "signature" in summary_data
