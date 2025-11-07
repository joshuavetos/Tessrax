import pytest

from tessrax.federation import FederationSimulator, ViewManager


def test_simulator_requires_byzantine_threshold():
    with pytest.raises(ValueError):
        FederationSimulator(["a", "b", "c"])  # insufficient nodes


def test_view_manager_timeout_rotation():
    manager = ViewManager(["a", "b", "c", "d"], view_timeout_ms=1)
    initial_leader = manager.leader()
    assert manager.maybe_advance(now=manager._view_start + 0.002)
    assert manager.leader() != initial_leader
    manager.force_advance(5)
    assert manager.leader() == manager.node_ids[5 % len(manager.node_ids)]
    with pytest.raises(ValueError):
        manager.force_advance(2)
