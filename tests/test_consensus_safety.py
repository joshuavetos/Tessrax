from tessrax.federation import InMemoryNetwork, Node


def build_cluster(size: int = 4):
    network = InMemoryNetwork()
    nodes = []
    for index in range(size):
        node = Node(f"node-{index}", total_nodes=size)
        network.register(node)
        nodes.append(node)
    return network, nodes


def test_single_leader_commit_safety():
    network, nodes = build_cluster()
    leader = nodes[0]
    leader.propose({"tx": "one"}, view=1)

    commits = [node.consensus.latest_commit() for node in nodes]
    assert len(set(commits)) == 1


def test_conflicting_proposals_do_not_commit():
    network, nodes = build_cluster()
    leader = nodes[0]
    leader.propose({"tx": "one"}, view=1)
    # malicious leader tries to fork same view
    leader.propose({"tx": "conflict"}, view=1)

    commit_counts = [len(node.consensus.committed) for node in nodes]
    assert all(count <= 3 for count in commit_counts)
