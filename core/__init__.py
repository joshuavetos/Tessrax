"""Legacy Tessrax core compatibility layer."""
from __future__ import annotations

from .contradiction_engine import ContradictionEngine
from .governance_kernel import (
    GovernanceKernel,
    classify_contradiction,
    route_to_governance_lane,
)
from .receipts import Receipt, write_receipt

__all__ = [
    "ContradictionEngine",
    "GovernanceKernel",
    "classify_contradiction",
    "route_to_governance_lane",
    "Receipt",
    "write_receipt",
]
