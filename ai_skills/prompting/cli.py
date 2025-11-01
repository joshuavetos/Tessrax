"""Command line interface for the hermetic prompt-engineering toolkit."""

from __future__ import annotations

import argparse
import json

from ai_skills.prompting.evaluator import Evaluator
from ai_skills.prompting.template_engine import TemplateEngine


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser with all sub-commands."""

    parser = argparse.ArgumentParser(
        prog="ai-skills",
        description="Render deterministic prompt templates and score model guesses.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render",
        help="Render a named template with the supplied context values.",
    )
    render_parser.add_argument(
        "--template", required=True, help="Template name to render."
    )
    render_parser.add_argument(
        "--task", required=True, help="Primary task description."
    )
    render_parser.add_argument(
        "--context", required=True, help="Supplementary context text."
    )
    render_parser.add_argument(
        "--var",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional key/value pairs to inject into the template.",
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Compare a model guess against the truth reference.",
    )
    score_parser.add_argument("--guess", required=True, help="Model guess to evaluate.")
    score_parser.add_argument(
        "--truth", required=True, help="Ground-truth reference text."
    )

    return parser


def _parse_vars(pairs: list[str] | None) -> dict[str, str]:
    """Convert KEY=VALUE pairs into a dictionary with validation."""

    parsed: dict[str, str] = {}
    if not pairs:
        return parsed
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Could not parse '{pair}'. Expected 'key=value'. Update the CLI arguments."
            )
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError("Encountered an empty key while parsing --var arguments.")
        parsed[key] = value
    return parsed


def render_command(args: argparse.Namespace) -> str:
    """Execute the ``render`` sub-command and return the rendered template."""

    engine = TemplateEngine()
    values = {"task": args.task, "context": args.context}
    values.update(_parse_vars(args.var))
    return engine.render(args.template, **values)


def score_command(args: argparse.Namespace) -> dict[str, str | float | bool]:
    """Execute the ``score`` sub-command and return a serialisable result dictionary."""

    evaluator = Evaluator()
    return evaluator.score_as_dict(args.guess, args.truth)


def main(argv: list[str] | None = None) -> int:
    """Entry-point compatible with ``python -m`` execution."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "render":
        rendered = render_command(args)
        print(rendered)
        return 0

    if args.command == "score":
        result = score_command(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    parser.error("No command specified.")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
