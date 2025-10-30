from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare(
    baseline: dict[str, object],
    candidate: dict[str, object],
    float_tolerance: float = 0.01,
) -> tuple[bool, list[str]]:
    discrepancies: list[str] = []

    base_meta = baseline.get("metadata", {})
    cand_meta = candidate.get("metadata", {})
    for key in sorted(set(base_meta) | set(cand_meta)):
        if key not in base_meta:
            discrepancies.append(f"metadata.{key} missing from baseline")
            continue
        if key not in cand_meta:
            discrepancies.append(f"metadata.{key} missing from candidate")
            continue
        if isinstance(base_meta[key], (int, float)):
            if abs(float(base_meta[key]) - float(cand_meta[key])) > float_tolerance:
                discrepancies.append(
                    f"metadata.{key} differs: {base_meta[key]} vs {cand_meta[key]}"
                )
        else:
            if base_meta[key] != cand_meta[key]:
                discrepancies.append(f"metadata.{key} differs: {base_meta[key]} vs {cand_meta[key]}")

    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    for key in sorted(set(base_metrics) | set(cand_metrics)):
        if key not in base_metrics or key not in cand_metrics:
            discrepancies.append(f"metrics.{key} missing between snapshots")
            continue
        if abs(float(base_metrics[key]) - float(cand_metrics[key])) > float_tolerance:
            discrepancies.append(f"metrics.{key} drifted: {base_metrics[key]} vs {cand_metrics[key]}")

    return not discrepancies, discrepancies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Tessrax env snapshots")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--float-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Emit warnings instead of failing when discrepancies are detected",
    )
    args = parser.parse_args(argv)

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    ok, discrepancies = compare(baseline, candidate, args.float_tolerance)
    if not ok:
        prefix = "::warning::" if args.warn_only else "::error::"
        print(f"{prefix}Snapshot divergence detected; manual re-audit required")
        for diff in discrepancies:
            print(diff)
        return 0 if args.warn_only else 1

    print("Snapshots consistent within tolerance")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI harness
    raise SystemExit(main())
