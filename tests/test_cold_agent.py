"""Integration test validating the Cold Agent benchmark pipeline."""

from __future__ import annotations

import json

from tessrax.cold_agent import bench


def test_cold_agent_bench_creates_receipt(tmp_path, monkeypatch):
    """Bench run should emit a governance-compliant receipt artifact."""

    # Redirect output path to a temporary directory to keep tests hermetic.
    output_path = tmp_path / "cold_agent_run_receipt.json"
    monkeypatch.setattr(bench, "OUTPUT_PATH", output_path)

    summary = bench.main()
    assert summary["status"] == "PASS"
    assert summary["integrity_score"] == 1.0
    assert summary["auditor"] == "Tessrax Governance Kernel v16"
    assert set(summary["clauses"]) == {"AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"}

    assert output_path.exists(), "Cold Agent receipt must be written to disk"
    payload = json.loads(output_path.read_text())
    assert payload["signature"] == summary["signature"]
    assert payload["audit_result"]["status"] is True
    assert payload["ledger_metadata"]["entries"] == summary["runtime_info"]["dataset_size"]
