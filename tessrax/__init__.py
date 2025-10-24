"""Tessrax ' research-grade governance toolkit."""

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
