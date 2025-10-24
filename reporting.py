"""Ledger reporting utilities for Tessrax-Core."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane


def _load_ledger(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                if isinstance(record, dict):
                    entries.append(record)
    return entries


def summarise_ledger(entries: Iterable[dict]) -> Dict[str, object]:
    config = load_config()
    lane_counts = Counter()
    for entry in entries:
        claims = entry.get("claims", [])
        stability = calculate_stability(claims)
        lane = route_to_governance_lane(stability, config.thresholds)
        lane_counts[lane] += 1
    total = sum(lane_counts.values())
    return {"total": total, "lane_counts": dict(lane_counts)}


def main() -> None:
    config = load_config()
    ledger_path = Path(config.logging.ledger_path)
    entries = _load_ledger(ledger_path)
    summary = summarise_ledger(entries)
    print(f"Ledger path: {ledger_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
