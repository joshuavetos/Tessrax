#!/usr/bin/env python3
"""Execute Tessrax module self-tests without relying on external pytest plugins."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tessrax.selftest import main as run_selftest_main  # noqa: E402


def main() -> int:
    """Entrypoint delegating to ``tessrax.selftest`` and ensuring exit integrity."""

    exit_code = run_selftest_main(sys.argv)
    if exit_code not in (0, 1):
        raise RuntimeError(f"Unexpected exit code {exit_code} from tessrax.selftest")
    return exit_code


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
