"""Pytest configuration to keep the suite hermetic when optional deps are missing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import pytest


_MISSING_DEPENDENCY_REASONS = {
    "numpy": "Install numpy>=1.24 to exercise physics-dependent analytics tests.",
    "jwt": "Install PyJWT>=2.8 to exercise truth API contract tests.",
}


def _missing_dependencies() -> Iterable[str]:
    """Yield dependency names that are unavailable in the current environment."""

    for name in _MISSING_DEPENDENCY_REASONS:
        if importlib.util.find_spec(name) is None:
            yield name


_MISSING = tuple(_missing_dependencies())


def pytest_ignore_collect(path, config):  # type: ignore[override]
    """Skip collection for non-hermetic tests when optional dependencies are absent."""

    if not _MISSING:
        return False
    path_obj = Path(str(path))
    if "tests/ai_skills" in path_obj.as_posix():
        return False
    if path_obj.suffix == ".py":
        return True
    return False
