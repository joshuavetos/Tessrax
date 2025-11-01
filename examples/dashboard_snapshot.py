"""Create a dashboard snapshot from simulated receipts."""

from __future__ import annotations

from pprint import pprint

from tessrax import (
    ClaimExtractor,
    ContradictionEngine,
    GovernanceKernel,
    Ledger,
    build_snapshot,
)


def main() -> None:
    raw_inputs = [
        {
            "claim_id": "trust-1",
            "subject": "Atlas City",
            "metric": "water_quality",
            "value": 0.92,
            "unit": "index",
            "timestamp": "2025-08-01T00:00:00Z",
            "source": "Environmental Agency",
        },
        {
            "claim_id": "trust-2",
            "subject": "Atlas City",
            "metric": "water_quality",
            "value": 0.74,
            "unit": "index",
            "timestamp": "2025-08-02T00:00:00Z",
            "source": "Civil Watch",
        },
        {
            "claim_id": "trust-3",
            "subject": "Atlas City",
            "metric": "water_quality",
            "value": 0.81,
            "unit": "index",
            "timestamp": "2025-08-03T00:00:00Z",
            "source": "Public Health",
        },
    ]

    extractor = ClaimExtractor(default_unit="index")
    claims = extractor.extract(raw_inputs)

    engine = ContradictionEngine(tolerance=0.07)
    contradictions = engine.detect(claims)

    kernel = GovernanceKernel()
    ledger = Ledger()

    for contradiction in contradictions:
        decision = kernel.process(contradiction)
        ledger.append(decision)

    snapshot = build_snapshot(ledger.receipts())
    print("Dashboard snapshot:")
    pprint(snapshot)


if __name__ == "__main__":
    main()
