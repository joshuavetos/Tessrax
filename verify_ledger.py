"""Verify Tessrax ledger consistency."""

from __future__ import annotations

import json
from pathlib import Path

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane


def verify_chain(path: str | None = None) -> tuple[bool, str | None]:
    config = load_config()
    ledger_path = Path(path) if path else Path(config.logging.ledger_path)
    if not ledger_path.exists():
        return False, f"Ledger not found: {ledger_path}"

    with ledger_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                return False, f"Invalid JSON at line {index}"

            records = payload if isinstance(payload, list) else [payload]
            for offset, record in enumerate(records, start=0):
                if not isinstance(record, dict):
                    continue
                claims = record.get("claims", [])
                computed_stability = calculate_stability(claims)
                expected_lane = route_to_governance_lane(
                    computed_stability, config.thresholds
                )

                stored_stability = record.get("stability_score")
                stored_lane = record.get("governance_lane")

                if (
                    stored_stability is not None
                    and abs(stored_stability - computed_stability) > 1e-6
                ):
                    return (
                        False,
                        f"Stability mismatch at line {index} (entry {offset + 1})",
                    )
                if stored_lane is not None and stored_lane != expected_lane:
                    return False, f"Lane mismatch at line {index} (entry {offset + 1})"
    return True, None


def main() -> None:
    ok, message = verify_chain()
    if not ok:
        raise SystemExit(message)
    print("Ledger verified successfully.")


if __name__ == "__main__":
    main()
