from __future__ import annotations

import json
from pathlib import Path

from tools import compare_env_snapshot


def _write_snapshot(path: Path, metadata: dict[str, object], metrics: dict[str, float]) -> None:
    payload = {"metadata": metadata, "metrics": metrics}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_env_snapshot_flags_differences(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_snapshot(baseline, {"build": "123"}, {"integrity": 0.9})
    _write_snapshot(candidate, {"build": "124"}, {"integrity": 0.9})

    exit_code = compare_env_snapshot.main([str(baseline), str(candidate)])

    assert exit_code == 1


def test_compare_env_snapshot_warn_only(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_snapshot(baseline, {"build": "123"}, {"integrity": 0.9})
    _write_snapshot(candidate, {"build": "124"}, {"integrity": 0.9})

    exit_code = compare_env_snapshot.main(
        [str(baseline), str(candidate), "--warn-only"]
    )

    assert exit_code == 0
