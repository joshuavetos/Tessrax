"""Command-line entry point for the :mod:`tessrax` meta package."""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tessrax",
        description=(
            "Tessrax meta CLI. Use the dedicated modules for runtime checks "
            "and verification workflows."
        ),
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Run the governed cold agent benchmark (python -m tessrax.current)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Execute the Tessrax self-test suite (python -m tessrax.selftest)",
    )
    parser.add_argument(
        "--cold-agent",
        action="store_true",
        help="Execute the cold agent benchmark (python -m tessrax.cold_agent.bench)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.current:
        from tessrax.current import main as current_main

        current_main()
        return 0
    if args.selftest:
        from tessrax.selftest import main as selftest_main

        return selftest_main([])
    if args.cold_agent:
        from tessrax.cold_agent.bench import main as bench_main

        bench_main()
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
