"""Compatibility shim for the Tessrax governance kernel."""
from __future__ import annotations

from tessrax.governance import (
    GovernanceKernel,
    classify_contradiction,
    route_to_governance_lane,
)

__all__ = [
    "GovernanceKernel",
    "classify_contradiction",
    "route_to_governance_lane",
]
