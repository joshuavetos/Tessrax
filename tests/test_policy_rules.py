"""Tests for Tessrax policy rule compilation and evaluation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tessrax.core import policy_rules


def test_compile_policy_rules_validates_payload() -> None:
    rules = policy_rules.compile_policy_rules(
        [
            {"id": "drift", "threshold": 0.25, "severity": "high"},
            {"id": "uptime", "threshold": 0.95, "severity": "medium"},
        ]
    )
    assert [rule.identifier for rule in rules] == ["drift", "uptime"]


def test_compile_policy_rules_rejects_bad_threshold() -> None:
    with pytest.raises(policy_rules.PolicyRuleError):
        policy_rules.compile_policy_rules([{"id": "drift", "threshold": 2.0}])


def test_evaluate_policy_rules_emits_receipt(tmp_path: Path) -> None:
    rules = policy_rules.compile_policy_rules([
        {"id": "drift", "threshold": 0.4, "severity": "high"},
    ])
    metrics = {"drift": 0.5}

    outcomes = policy_rules.evaluate_policy_rules(rules, metrics, receipt_dir=tmp_path)

    assert outcomes == {"drift": True}
    receipt_path = tmp_path / "policy_rule_evaluation_receipt.json"
    assert receipt_path.exists()
    payload = receipt_path.read_text(encoding="utf-8")
    assert "policy_rule_evaluation" in payload
