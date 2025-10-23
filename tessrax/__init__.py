"""Tessrax – research-grade governance toolkit."""

from .extraction import ClaimExtractor
from .contradiction import ContradictionEngine
from .governance import GovernanceKernel
from .ledger import Ledger, LedgerReceipt
from .dashboard import build_snapshot
from .types import Claim, ContradictionRecord, GovernanceDecision

__all__ = [
    "ClaimExtractor",
    "ContradictionEngine",
    "GovernanceKernel",
    "Ledger",
    "LedgerReceipt",
    "Claim",
    "ContradictionRecord",
    "GovernanceDecision",
    "build_snapshot",
]

__version__ = "0.1.0"
