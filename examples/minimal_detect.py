"""Minimal contradiction detection demo."""

from __future__ import annotations

from pprint import pprint

from tessrax import ClaimExtractor, ContradictionEngine


def main() -> None:
    raw_inputs = [
        {
            "claim_id": "energy-1",
            "subject": "Aurora Station",
            "metric": "emissions",
            "value": 42.0,
            "unit": "ktCO2e",
            "timestamp": "2025-10-17T13:10:00Z",
            "source": "Operations",
        },
        {
            "claim_id": "energy-2",
            "subject": "Aurora Station",
            "metric": "emissions",
            "value": 30.0,
            "unit": "ktCO2e",
            "timestamp": "2025-10-18T13:10:00Z",
            "source": "Sustainability",
        },
        {
            "claim_id": "energy-3",
            "subject": "Aurora Station",
            "metric": "emissions",
            "value": 31.0,
            "unit": "ktCO2e",
            "timestamp": "2025-10-18T16:10:00Z",
            "source": "Audit",
        },
    ]

    extractor = ClaimExtractor(default_unit="ktCO2e")
    claims = extractor.extract(raw_inputs)

    engine = ContradictionEngine(tolerance=0.05)
    contradictions = engine.detect(claims)

    print("Detected contradictions:")
    for record in contradictions:
        pprint(
            {
                "pair": [record.claim_a.claim_id, record.claim_b.claim_id],
                "severity": record.severity,
                "delta": round(record.delta, 3),
                "reasoning": record.reasoning,
            }
        )


if __name__ == "__main__":
    main()
