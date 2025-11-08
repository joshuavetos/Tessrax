"""CLI harness for :mod:`tessrax.ledger`."""

from __future__ import annotations

from typing import Sequence

from . import _cli


def main(argv: Sequence[str] | None = None) -> None:
    _cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main(None)
