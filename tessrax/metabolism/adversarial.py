"""
Adversarial Metabolism Engine — Generates synthetic contradictions to probe reconciliation limits.
Emits ledger event_type 'ADVERSARIAL_CONTRADICTION'.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import uuid
from typing import Dict, List

DEFAULT_SEED = 1337
MAX_BATCH = 10


class AdversarialAgent:
    def __init__(self, seed: int | None = DEFAULT_SEED, max_batch: int = MAX_BATCH):
        self.rng = random.Random(seed)
        self.max_batch = max_batch

    def synthesize_contradiction(self) -> Dict:
        A = self.rng.choice(
            [
                "All governance is self-consistent",
                "No governance can be self-consistent",
                "Every truth can be proven",
                "Some truths cannot be proven",
                "Freedom requires structure",
                "Structure erodes freedom",
            ]
        )
        B = self.rng.choice(
            [
                "Truth is recursive",
                "Truth is terminal",
                "Governance must evolve",
                "Governance must stabilize",
            ]
        )
        return {
            "id": f"adv-{uuid.uuid4().hex[:8]}",
            "event_type": "ADVERSARIAL_CONTRADICTION",
            "A": A,
            "B": B,
            "confidence": round(self.rng.uniform(0.6, 0.95), 3),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "source": "adversarial_generator",
            "meta": {"synthetic": True},
        }

    def run_batch(self, n: int = 5) -> List[Dict]:
        n = min(n, self.max_batch)
        return [self.synthesize_contradiction() for _ in range(n)]

    def generate(self, current_integrity: float) -> List[Dict]:
        budget = compute_adversarial_budget(current_integrity, self.max_batch)
        return self.run_batch(budget)


def compute_adversarial_budget(
    integrity: float,
    max_batch: int,
    alpha: float = 2.0,
) -> int:
    if integrity >= 0.999:
        return max_batch
    denom = max(1.0 - integrity, 1e-6)
    raw = alpha / denom
    allowed = min(max_batch, int(raw))
    return max(0, allowed)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate adversarial contradiction batches")
    parser.add_argument("--batch", type=int, default=5, help="Number of contradictions to generate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed override")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=MAX_BATCH,
        help="Upper bound for synthetic contradiction generation",
    )
    parser.add_argument(
        "--integrity",
        type=float,
        default=0.93,
        help="Current system integrity score for adversarial budgeting",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)
    agent = AdversarialAgent(seed=args.seed, max_batch=args.max_batch)
    requested = min(args.batch, args.max_batch)
    integrity_limited_budget = compute_adversarial_budget(args.integrity, requested)
    batch = agent.run_batch(integrity_limited_budget)
    for record in batch:
        json.dump(record, sys.stdout)
        sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
