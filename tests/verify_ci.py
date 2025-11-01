"""Utility to capture CI collection output for coherence audits."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "ci_runs.log"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    root = Path(__file__).resolve().parent.parent
    audit_kernel_src = root / "packages" / "audit-kernel" / "src"
    python_path = os.environ.get("PYTHONPATH", "")
    extra_path = os.pathsep.join(
        filter(None, [str(root), str(audit_kernel_src), python_path])
    )
    env = dict(os.environ, PYTHONPATH=extra_path)
    result = subprocess.run(
        ["pytest", "--collect-only"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    header = f"## CI verification run @ {timestamp}\n"
    footer = f"\nexit_code={result.returncode}\n"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(result.stdout)
        if result.stderr:
            handle.write("\n[stderr]\n")
            handle.write(result.stderr)
        handle.write(footer)
    if result.returncode != 0:
        raise SystemExit("Pytest collection failed")
    print(f"CI verification log appended to {LOG_FILE}")


if __name__ == "__main__":
    main()
