"""Tests for the Tessrax federated test harness."""

from __future__ import annotations

import asyncio

from federation.test_harness import simulate_federation


def test_byzantine_detection() -> None:
    result = asyncio.run(simulate_federation())
    assert result["byzantine_detected"] is True
    assert isinstance(result["consensus_root"], str)
