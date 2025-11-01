from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = [
    re.compile(r"STRIPE_[A-Za-z0-9]{10,}"),
    re.compile(r"JWT_[A-Za-z0-9]{10,}"),
    re.compile(r"HMAC_[A-Za-z0-9]{10,}"),
]

EXCLUDED_DIRS = {
    Path("docs"),
    Path("tests/fixtures"),
}


def load_ignore(path: Path) -> set[str]:
    ignores: set[str] = set()
    if not path.exists():
        return ignores
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ignores.add(line)
    return ignores


def should_skip(file_path: Path, ignores: set[str]) -> bool:
    if any(part in EXCLUDED_DIRS for part in file_path.parents):
        return True
    return any(file_path.match(pattern) for pattern in ignores)


def scan(root: Path, ignores: set[str]) -> list[str]:
    problems: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if should_skip(relative, ignores):
            continue
        if relative.name == ".env" or relative.suffix == ".env":
            problems.append(f"{relative} committed .env file")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                token = match.group(0)
                if token.endswith("TEST") or token.endswith("PLACEHOLDER"):
                    continue
                problems.append(
                    f"{relative}:{match.start()} potential secret '{token}'"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lint Tessrax repository for leaked secrets"
    )
    parser.add_argument("path", nargs="?", default=Path("."), type=Path)
    args = parser.parse_args()

    ignores = load_ignore(Path(".secrets-lint-ignore"))
    problems = scan(args.path.resolve(), ignores)
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print("No secrets detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
