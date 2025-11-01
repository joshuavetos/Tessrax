"""Tests for the central Tessrax engine."""

from __future__ import annotations

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane


def _claims(values: list[str]) -> list[dict]:
    return [
        {"agent": f"agent-{idx}", "claim": value}
        for idx, value in enumerate(values, start=1)
    ]


def test_calculate_stability_high_consensus() -> None:
    claims = _claims(["A", "A", "A"])
    assert calculate_stability(claims) == 1.0


def test_calculate_stability_low_consensus() -> None:
    claims = _claims(["A", "B", "C"])
    assert calculate_stability(claims) == 0.0


def test_route_to_governance_lane_thresholds() -> None:
    config = load_config()
    assert route_to_governance_lane(0.9, config.thresholds) == "autonomic"
    assert route_to_governance_lane(0.6, config.thresholds) == "deliberative"
    assert route_to_governance_lane(0.1, config.thresholds) == "behavioral_audit"
