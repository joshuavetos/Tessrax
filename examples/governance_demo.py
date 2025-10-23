"""Simulate a governance cycle and emit ledger receipts."""

from __future__ import annotations

from pathlib import Path

from tessrax import (
    ClaimExtractor,
    ContradictionEngine,
    GovernanceKernel,
    Ledger,
)


def main() -> None:
    raw_inputs = [
        {
            "claim_id": "budget-1",
            "subject": "Helios Program",
            "metric": "spend",
            "value": 12_000_000,
            "unit": "USD",
            "timestamp": "2025-09-01T00:00:00Z",
            "source": "Finance",
        },
        {
            "claim_id": "budget-2",
            "subject": "Helios Program",
            "metric": "spend",
            "value": 9_100_000,
            "unit": "USD",
            "timestamp": "2025-09-02T00:00:00Z",
            "source": "Program Office",
        },
    ]

    extractor = ClaimExtractor(default_unit="USD")
    claims = extractor.extract(raw_inputs)

    engine = ContradictionEngine(tolerance=0.08)
    contradictions = engine.detect(claims)

    kernel = GovernanceKernel()
    ledger = Ledger()

    for contradiction in contradictions:
        decision = kernel.process(contradiction)
        receipt = ledger.append(decision)
        print(f"Recorded {receipt.hash[:8]} for action {decision.action} at severity {contradiction.severity}")

    export_path = Path("artifacts/ledger_demo.jsonl")
    ledger.export(export_path)
    print(f"Ledger exported to {export_path.relative_to(Path.cwd())}")

    ledger.verify()  # Raises on failure
    print("Ledger integrity verified.")


if __name__ == "__main__":
    main()
