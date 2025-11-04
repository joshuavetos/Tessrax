"""Ethical drift simulator regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import sandbox.ethical_drift as ethical_drift


def test_simulate_ethics_is_reproducible(tmp_path: Path) -> None:
    receipts_path = tmp_path / "ethical_drift_receipts.jsonl"
    summary_path = tmp_path / "ethical_drift_summary.json"

    original_receipts = ethical_drift.RECEIPTS_PATH
    original_summary = ethical_drift.SUMMARY_PATH
    ethical_drift.RECEIPTS_PATH = receipts_path
    ethical_drift.SUMMARY_PATH = summary_path

    result_first = result_second = None
    receipts_first = receipts_second = ""
    try:
        result_first = ethical_drift.simulate_ethics(conflicts=5, seed=42)
        receipts_first = receipts_path.read_text(encoding="utf-8")

        result_second = ethical_drift.simulate_ethics(conflicts=5, seed=42)
        receipts_second = receipts_path.read_text(encoding="utf-8")
    finally:
        ethical_drift.RECEIPTS_PATH = original_receipts
        ethical_drift.SUMMARY_PATH = original_summary

    assert receipts_first == receipts_second
    assert result_first is not None and result_second is not None
    assert result_first["mean_drift"] == result_second["mean_drift"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["event_type"] == "ETHICAL_DRIFT_SUMMARY"
