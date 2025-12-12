"""Governance utilities exposing the TIP registry and validation CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from . import tip_registry, tip_validate


def _load_legacy_governance_module():
    module_path = Path(__file__).resolve().parent.parent / "governance.py"
    spec = importlib.util.spec_from_file_location("tessrax._governance_legacy", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive import guard
        raise ImportError(f"Unable to load governance module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_governance_module()
route_to_governance_lane = _legacy.route_to_governance_lane
classify_contradiction = _legacy.classify_contradiction
GovernanceKernel = _legacy.GovernanceKernel

__all__ = [
    "tip_registry",
    "tip_validate",
    "route_to_governance_lane",
    "classify_contradiction",
    "GovernanceKernel",
]
