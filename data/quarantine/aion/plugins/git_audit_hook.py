"""Git hook integration for the Tessrax AION audit engine.

The hook can be installed as a pre-commit script. It executes the audit
engine and blocks the commit if the integrity requirements fail.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_audit() -> int:
    """Invoke the audit engine as a subprocess and return its exit code."""
    executable = Path(__file__).resolve().parents[1] / "bin" / "aion-audit"
    if not executable.exists():
        raise FileNotFoundError(f"Expected audit executable at {executable}")
    result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def main() -> int:
    """Entry point used by Git hooks."""
    code = run_audit()
    if code != 0:
        sys.stderr.write("TESS MODE BLOCK: AION audit failed; commit aborted.\n")
    return code


if __name__ == "__main__":  # pragma: no cover - CLI behavior
    raise SystemExit(main())
