from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

METRICS_PATH = ROOT / "tessrax" / "metrics" / "epistemic_health.py"
spec = importlib.util.spec_from_file_location(
    "tessrax.metrics.epistemic_health", METRICS_PATH
)
epistemic_health = importlib.util.module_from_spec(spec)
assert spec and spec.loader  # narrow typing for mypy/linters
spec.loader.exec_module(epistemic_health)

SNAPSHOT_PATH = Path("out/env_snapshot.json")


def _pip_freeze() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def build_snapshot(notes: str | None = None) -> dict[str, object]:
    frozen = _pip_freeze().splitlines()
    env_hash = hashlib.sha256("\n".join(sorted(frozen)).encode("utf-8")).hexdigest()

    integrity = epistemic_health.compute_integrity([0.18, 0.21, 0.2, 0.19])
    drift = epistemic_health.compute_drift([(0.0, 0.2), (1.0, 0.22), (2.0, 0.215)])
    severity = epistemic_health.compute_severity([0.19, 0.2, 0.21], [0.21, 0.19, 0.2])
    independence = epistemic_health.compute_entropy(
        ["alpha", "beta", "beta", "gamma", "gamma"]
    )

    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "environment_hash": env_hash,
    }
    if notes:
        metadata["notes"] = notes

    return {
        "metadata": metadata,
        "metrics": {
            "integrity": round(integrity, 4),
            "drift": round(drift, 4),
            "severity": round(severity, 4),
            "independence": round(independence, 4),
        },
    }


def write_snapshot(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Tessrax environment snapshot"
    )
    parser.add_argument(
        "--output", type=Path, default=SNAPSHOT_PATH, help="Destination file"
    )
    parser.add_argument(
        "--notes", type=str, default=None, help="Optional override notes"
    )
    args = parser.parse_args(argv)

    payload = build_snapshot(notes=args.notes)
    write_snapshot(args.output, payload)
    print(f"Snapshot written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
