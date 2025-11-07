from tessrax.federation import FederationSimulator


def test_simulator_reaches_consensus_and_latency_bound():
    simulator = FederationSimulator(["a", "b", "c", "d"])  # four nodes minimum
    for index in range(3):
        simulator.run_round({"height": index})
    assert simulator.consensus_reached()
    assert simulator.average_latency() <= 150.0
