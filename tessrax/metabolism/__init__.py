"""Metabolic subsystems for Tessrax."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "AdversarialAgent": ("tessrax.metabolism.adversarial", "AdversarialAgent"),
    "AsyncContradictionDetector": (
        "tessrax.metabolism.async_detector",
        "AsyncContradictionDetector",
    ),
    "ReconciliationEngine": ("tessrax.metabolism.reconcile", "ReconciliationEngine"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr = _EXPORTS[name]
    module: ModuleType = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - interactive convenience
    return sorted(set(globals()) | set(_EXPORTS))
