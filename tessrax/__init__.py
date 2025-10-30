"""Tessrax ' research-grade governance toolkit."""

from __future__ import annotations

import sys
import warnings

if sys.version_info < (3, 11):  # pragma: no cover - executed during import
    warnings.warn(
        "Tessrax v16 targets Python 3.11+. Use the v15-stable tag for Python 3.10 deployments.",
        DeprecationWarning,
        stacklevel=2,
    )

from tessrax.contradiction import ContradictionEngine
from tessrax.dashboard import build_snapshot
from tessrax.extraction import ClaimExtractor
from tessrax.governance import GovernanceKernel
from tessrax.ledger import Ledger, LedgerReceipt
from tessrax.metabolism.reconcile import ReconciliationEngine
from tessrax.types import Claim, ContradictionRecord, GovernanceDecision

__all__ = [
    "ClaimExtractor",
    "ContradictionEngine",
    "GovernanceKernel",
    "Ledger",
    "LedgerReceipt",
    "ReconciliationEngine",
    "Claim",
    "ContradictionRecord",
    "GovernanceDecision",
    "build_snapshot",
]

__version__ = "0.1.0"
