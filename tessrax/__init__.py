"""Tessrax ' research-grade governance toolkit."""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType
from typing import Any, Dict, Tuple

if sys.version_info < (3, 11):  # pragma: no cover - executed during import
    warnings.warn(
        "Tessrax v16 targets Python 3.11+. Use the v15-stable tag for Python 3.10 deployments.",
        DeprecationWarning,
        stacklevel=2,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ClaimExtractor": ("tessrax.extraction", "ClaimExtractor"),
    "ContradictionEngine": ("tessrax.contradiction", "ContradictionEngine"),
    "GovernanceKernel": ("tessrax.governance", "GovernanceKernel"),
    "Ledger": ("tessrax.ledger", "Ledger"),
    "LedgerReceipt": ("tessrax.ledger", "LedgerReceipt"),
    "ReconciliationEngine": ("tessrax.metabolism.reconcile", "ReconciliationEngine"),
    "Claim": ("tessrax.types", "Claim"),
    "ContradictionRecord": ("tessrax.types", "ContradictionRecord"),
    "GovernanceDecision": ("tessrax.types", "GovernanceDecision"),
    "build_snapshot": ("tessrax.dashboard", "build_snapshot"),
}

__all__ = sorted(_EXPORTS)
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazily import Tessrax submodules to keep top-level import lightweight."""

    if name not in _EXPORTS:
        raise AttributeError(f"module 'tessrax' has no attribute {name!r}") from None
    module_name, attribute = _EXPORTS[name]
    module: ModuleType = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - interactive convenience
    return sorted(set(globals()) | set(_EXPORTS))
