"""Regression tests for the Tessrax federated node demo."""

from __future__ import annotations

import asyncio

from federation.demo_federation import federated_run


def test_federated_consensus() -> None:
    """The federation demo should achieve deterministic consensus."""

    assert asyncio.run(federated_run()) is True
