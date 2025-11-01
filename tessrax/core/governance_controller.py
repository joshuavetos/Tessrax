"""Command-line governance audit orchestrator for Tessrax.

This module implements an executable governance audit routine that integrates
field evidence, validates policy alignment metrics, records receipts, and
emits structured compliance reports. The controller is designed for use in
local development hooks, CI pipelines, and continuous watchdog monitoring.
"""

from __future__ import annotations

import importlib.machinery
import json
import sys
import traceback
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_tessrax_package_stub() -> None:
    if "tessrax" in sys.modules:
        return
    tessrax_stub = types.ModuleType("tessrax")
    tessrax_stub.__path__ = [str(PROJECT_ROOT / "tessrax")]
    tessrax_stub.__package__ = "tessrax"
    tessrax_stub.__file__ = str(PROJECT_ROOT / "tessrax" / "__init__.py")
    tessrax_stub.__spec__ = importlib.machinery.ModuleSpec(
        "tessrax", loader=None, is_package=True
    )
    tessrax_stub.__spec__.submodule_search_locations = [str(PROJECT_ROOT / "tessrax")]
    sys.modules["tessrax"] = tessrax_stub


try:
    import tessrax  # type: ignore[import]
except Exception:
    _ensure_tessrax_package_stub()
else:
    if (
        not hasattr(tessrax, "__spec__")
        or getattr(tessrax.__spec__, "submodule_search_locations", None) is None
    ):
        _ensure_tessrax_package_stub()

from tessrax.core.governance_kernel import GovernanceKernel

AUDIT_OUTPUT_PATH = Path("out/governance_audit.json")
AUDIT_RECEIPTS_PATH = Path("out/governance_audit_receipts.jsonl")


@dataclass(slots=True)
class AuditCheck:
    """Structured representation of an audit assertion."""

    name: str
    passed: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representation."""

        return {"name": self.name, "passed": self.passed, "details": self.details}


class GovernanceController:
    """Execute governance audits and persist receipts."""

    def __init__(
        self,
        *,
        output_path: Path = AUDIT_OUTPUT_PATH,
        receipts_path: Path = AUDIT_RECEIPTS_PATH,
    ) -> None:
        self.output_path = output_path
        self.receipts_path = receipts_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipts_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Run the governance audit and return the structured result."""

        kernel = GovernanceKernel(auto_integrate=True)
        summary = kernel.integrate_field_evidence(refresh=False)
        checks = self._evaluate_summary(summary)
        status = "pass" if all(check.passed for check in checks) else "fail"
        report = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "checks": [check.as_dict() for check in checks],
        }
        self._write_report(report)
        self._append_receipt(report)
        return report

    def _evaluate_summary(self, summary: dict[str, Any]) -> list[AuditCheck]:
        """Create audit checks based on field evidence summary."""

        checks: list[AuditCheck] = []
        total_records = summary.get("total_records", 0)
        checks.append(
            AuditCheck(
                name="field_evidence_loaded",
                passed=bool(total_records and total_records > 0),
                details={"total_records": total_records},
            )
        )
        alignment_scores = summary.get("alignment_scores", {})
        numeric_scores = [
            score
            for score in alignment_scores.values()
            if isinstance(score, (int, float))
        ]
        score_bounds_passed = bool(numeric_scores) and all(
            0.0 <= float(score) <= 1.0 for score in numeric_scores
        )
        checks.append(
            AuditCheck(
                name="alignment_scores_within_bounds",
                passed=score_bounds_passed,
                details={
                    "count": len(numeric_scores),
                    "min": min(numeric_scores) if numeric_scores else None,
                    "max": max(numeric_scores) if numeric_scores else None,
                },
            )
        )
        contradiction_cases = summary.get("contradiction_cases", [])
        checks.append(
            AuditCheck(
                name="contradiction_cases_indexed",
                passed=isinstance(contradiction_cases, list),
                details={
                    "contradiction_count": (
                        len(contradiction_cases)
                        if isinstance(contradiction_cases, list)
                        else None
                    )
                },
            )
        )
        category_counts = summary.get("category_counts", {})
        well_structured_categories = isinstance(category_counts, dict) and all(
            isinstance(key, str) and isinstance(value, int) and value >= 0
            for key, value in category_counts.items()
        )
        checks.append(
            AuditCheck(
                name="category_distribution_valid",
                passed=well_structured_categories,
                details={
                    "category_count": (
                        len(category_counts)
                        if isinstance(category_counts, dict)
                        else None
                    )
                },
            )
        )
        return checks

    def _write_report(self, report: dict[str, Any]) -> None:
        """Persist the audit report to the configured output path."""

        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _append_receipt(self, report: dict[str, Any]) -> None:
        """Append a receipt entry capturing the audit outcome."""

        receipt = {
            "timestamp": report["timestamp"],
            "status": report["status"],
            "checks": [
                check["name"]
                for check in report.get("checks", [])
                if check.get("passed")
            ],
        }
        with self.receipts_path.open("a", encoding="utf-8") as handle:
            json.dump(receipt, handle)
            handle.write("\n")


def _main() -> int:
    """Entry point for command-line execution."""

    controller = GovernanceController()
    try:
        result = controller.run()
    except Exception as exc:  # pragma: no cover - defensive logging for CLI usage
        traceback.print_exc()
        print(
            f"[Tessrax] Governance audit encountered an error: {exc}", file=sys.stderr
        )
        return 1
    status = result.get("status")
    if status != "pass":
        print("[Tessrax] Governance audit failed.", file=sys.stderr)
        return 1
    print("[Tessrax] Governance audit completed successfully.")
    return 0


def _self_test() -> bool:
    """Run a lightweight self-test of the governance controller."""

    controller = GovernanceController(
        output_path=Path("out/_self_test_governance_audit.json"),
        receipts_path=Path("out/_self_test_receipts.jsonl"),
    )
    result = controller.run()
    assert result["status"] == "pass", "Self-test governance audit should pass"
    return True


if __name__ == "__main__":
    sys.exit(_main())
