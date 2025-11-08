"""Lazy adapter for the vendored :mod:`audit_kernel` package.

The adapter keeps Tessrax core importable in cold environments by resolving
and loading the bundled ``packages/audit-kernel`` distribution on demand. The
module performs runtime verification (``RVC-001``) for every resolution so
failures surface with actionable diagnostics rather than silent fallbacks.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

__all__ = ["load", "__getattr__", "__dir__"]

_VENDOR_RELATIVE = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "audit-kernel"
    / "src"
)

_cached_module: ModuleType | None = None


def load() -> ModuleType:
    """Return the vendored :mod:`audit_kernel` module, importing it lazily."""

    global _cached_module
    if _cached_module is not None:
        return _cached_module

    search_path = str(_VENDOR_RELATIVE)
    if _VENDOR_RELATIVE.exists() and search_path not in sys.path:
        sys.path.insert(0, search_path)
    try:
        module = import_module("audit_kernel")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError(
            "Vendored audit_kernel package not found; expected at "
            f"{_VENDOR_RELATIVE}"
        ) from exc
    _cached_module = module
    return module


def __getattr__(name: str) -> Any:
    """Proxy attribute access to the vendored module with lazy loading."""

    module = load()
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(
            f"module 'audit_kernel' has no attribute {name!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - interactive convenience
    module = load()
    return sorted(set(globals()) | set(dir(module)))

