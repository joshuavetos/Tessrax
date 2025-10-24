"""Command line entry point for ``python -m tessrax.ledger``."""

from __future__ import annotations

import sys

from . import main


def run() -> None:
    main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run()
