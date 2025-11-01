"""Epistemic health metrics utilities."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

_BASELINE_FILE = pathlib.Path("integrity_baseline.json")


def update_baseline(current_integrity: float) -> list[dict[str, Any]]:
    base = _BASELINE_FILE
    data = {"timestamp": time.time(), "integrity": current_integrity}
    hist = json.loads(base.read_text()) if base.exists() else []
    hist.append(data)
    hist = hist[-7:]
    base.write_text(json.dumps(hist, indent=2))
    return hist


def delta_integrity() -> float:
    base = _BASELINE_FILE
    if not base.exists():
        return 0.0
    hist = json.loads(base.read_text())
    if len(hist) < 2:
        return 0.0
    return hist[-1]["integrity"] - hist[-2]["integrity"]


def main() -> None:
    history = update_baseline(1.0)
    print(json.dumps(history, indent=2))
    print("Delta:", delta_integrity())


if __name__ == "__main__":
    main()
