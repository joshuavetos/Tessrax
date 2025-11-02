"""ETL pipeline that converts CDC influenza surveillance into Tessrax claims."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_DATASET = Path("data/public_health/influenza_activity.csv")
DEFAULT_ENDPOINT = "http://localhost:8000"


@dataclass
class InfluenzaRecord:
    """Row of influenza activity data."""

    state: str
    week_start: str
    ili_percentage: float
    lab_confirmed_flu_cases: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "InfluenzaRecord":
        try:
            return cls(
                state=row["state"].strip(),
                week_start=row["week_start"].strip(),
                ili_percentage=float(row["ili_percentage"]),
                lab_confirmed_flu_cases=int(row["lab_confirmed_flu_cases"]),
            )
        except KeyError as exc:  # pragma: no cover - data contract violation
            raise ValueError(f"Missing column in dataset: {exc}") from exc


def load_dataset(path: Path) -> list[InfluenzaRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [InfluenzaRecord.from_row(row) for row in reader]
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def to_claims(records: Iterable[InfluenzaRecord]) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    for record in records:
        stability_context = {
            "state": record.state,
            "week_start": record.week_start,
            "ili_percentage": record.ili_percentage,
            "lab_confirmed_flu_cases": record.lab_confirmed_flu_cases,
        }
        claims.append(
            {
                "agent": "etl.public_health",
                "claim": (
                    f"{record.state} reported {record.lab_confirmed_flu_cases} lab-confirmed "
                    f"influenza cases for week starting {record.week_start} with ILI "
                    f"activity at {record.ili_percentage:.1f}%"
                ),
                "context": stability_context,
            }
        )
    if not claims:
        raise ValueError("No claims generated from dataset")
    return claims


def submit_claims(claims: list[dict[str, object]], endpoint: str) -> dict[str, object]:
    response = requests.post(
        f"{endpoint.rstrip('/')}/submit_claims",
        json={"claims": claims},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("API response must be a JSON object")
    return payload


def run_etl(dataset: Path, endpoint: str, dry_run: bool) -> None:
    records = load_dataset(dataset)
    claims = to_claims(records)
    if dry_run:
        for claim in claims:
            print(claim)
        return
    result = submit_claims(claims, endpoint)
    print("Submission succeeded:", result)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Influenza ETL for Tessrax cluster")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="CSV dataset path")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="Governance API base URL")
    parser.add_argument("--dry-run", action="store_true", help="Print claims without submitting")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    run_etl(args.dataset, args.endpoint, args.dry_run)


if __name__ == "__main__":
    main()
