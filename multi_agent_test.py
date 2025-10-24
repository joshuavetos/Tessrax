"""Run a consistency check across a cluster of agents."""

from __future__ import annotations

from typing import List

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane


def generate_sample_claims() -> List[dict]:
    return [
        {"agent": "alpha", "claim": "System operating within normal bounds."},
        {"agent": "beta", "claim": "System operating within normal bounds."},
        {"agent": "gamma", "claim": "System operating within normal bounds."},
        {"agent": "delta", "claim": "System requires review."},
    ]


def run() -> None:
    config = load_config()
    claims = generate_sample_claims()
    stability = calculate_stability(claims)
    lane = route_to_governance_lane(stability, config.thresholds)

    print("Evaluated claims:")
    for claim in claims:
        print(f" - {claim['agent']}: {claim['claim']}")

    print(f"\nStability score: {stability:.3f}")
    print(f"Governance lane: {lane}")


if __name__ == "__main__":
    run()
