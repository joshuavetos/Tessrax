from __future__ import annotations

import sys

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 11


def main() -> int:
    if sys.version_info[:2] != (REQUIRED_MAJOR, REQUIRED_MINOR):
        print(
            "Pre-commit must run under Python 3.11.x."
            f" Detected {sys.version_info.major}.{sys.version_info.minor}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
