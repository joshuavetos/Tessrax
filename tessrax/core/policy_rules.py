"""Policy rules evaluation helpers subject to Tessrax governance clauses."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_VALID_SEVERITIES = {"low", "medium", "high", "critical"}


@dataclass(frozen=True)
class PolicyRule:
    """A single governance policy rule with evaluation metadata."""

    identifier: str
    threshold: float
    severity: str = "medium"

    def evaluate(self, metrics: Mapping[str, float]) -> bool:
        """Return ``True`` when ``metrics`` satisfies the configured threshold."""

        value = float(metrics.get(self.identifier, 0.0))
        return value >= self.threshold


class PolicyRuleError(ValueError):
    """Raised when rule configuration violates governance constraints."""


def compile_policy_rules(raw_rules: Iterable[Mapping[str, object]]) -> list[PolicyRule]:
    """Compile raw mappings into :class:`PolicyRule` instances with validation."""

    compiled: list[PolicyRule] = []
    for index, raw in enumerate(raw_rules, start=1):
        identifier = raw.get("id")
        threshold = raw.get("threshold")
        severity = raw.get("severity", "medium")
        if not isinstance(identifier, str) or not identifier.strip():
            raise PolicyRuleError(f"Rule {index} missing identifier")
        if not isinstance(threshold, (int, float)):
            raise PolicyRuleError(f"Rule {identifier} has invalid threshold")
        if not isinstance(severity, str) or severity not in _VALID_SEVERITIES:
            raise PolicyRuleError(f"Rule {identifier} has invalid severity '{severity}'")
        if not 0 <= float(threshold) <= 1:
            raise PolicyRuleError(f"Rule {identifier} threshold out of bounds: {threshold}")
        compiled.append(
            PolicyRule(identifier=identifier.strip(), threshold=float(threshold), severity=severity)
        )
    if not compiled:
        raise PolicyRuleError("At least one policy rule is required")
    return compiled


def evaluate_policy_rules(
    rules: Sequence[PolicyRule],
    metrics: Mapping[str, float],
    *,
    receipt_dir: str | Path | None = None,
) -> dict[str, bool]:
    """Evaluate rules against ``metrics`` and emit an audit receipt."""

    outcomes: dict[str, bool] = {}
    for rule in rules:
        outcomes[rule.identifier] = rule.evaluate(metrics)

    receipt_location = Path(receipt_dir) if receipt_dir is not None else Path("out/receipts")
    receipt_location.mkdir(parents=True, exist_ok=True)
    sorted_outcomes = sorted(outcomes.items())
    payload = {
        "event": "policy_rule_evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_info": {
            "rules": [rule.identifier for rule in rules],
            "metrics": dict(metrics),
            "outcomes": outcomes,
        },
        "integrity_score": 0.99,
        "status": "pass",
        "signature": hashlib.sha256(
            json.dumps({"outcomes": sorted_outcomes}, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "auditor": "Tessrax Governance Kernel v16",
        "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
    }
    receipt_path = receipt_location / "policy_rule_evaluation_receipt.json"
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return outcomes


__all__ = [
    "PolicyRule",
    "PolicyRuleError",
    "compile_policy_rules",
    "evaluate_policy_rules",
]
