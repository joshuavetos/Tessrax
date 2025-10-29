"""Ensure dependency lock files remain synchronized."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Sequence


ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements-lock.txt"


def _frozen_dependencies() -> str:
    """Return the canonical dependency snapshot without contourpy."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if "contourpy" not in line]
    if lines and lines[-1]:
        lines.append("")
    return "\n".join(lines)


def _read(path: pathlib.Path) -> str:
    return path.read_text() if path.exists() else ""


def verify() -> bool:
    expected = _frozen_dependencies()
    req = _read(REQUIREMENTS)
    lock = _read(LOCK)
    matches = True

    if req != expected:
        print("requirements.txt is out of sync with the current environment.")
        matches = False
    if lock != expected:
        print("requirements-lock.txt is out of sync with the current environment.")
        matches = False
    if matches:
        print("✅ Dependency snapshots are synchronized.")
    return matches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify dependency synchronization")
    args = parser.parse_args(argv)

    if args.verify:
        return 0 if verify() else 1

    parser.error("No action specified. Use --verify to check dependency synchronization.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
