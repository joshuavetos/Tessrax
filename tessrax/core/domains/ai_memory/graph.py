"""
memory_contradiction_graph.py
Visualizes coherence vs. contradiction density across AI memory snapshots.
"""

import matplotlib.pyplot as plt
from memory_primitives import conflict_density, coherence_penalty
from memory_contradiction_detector import detect_memory_conflicts

def plot_memory_contradictions(total=100, conflicts=None):
    if conflicts is None:
        conflicts = detect_memory_conflicts()
    conflict_count = len(conflicts)
    dens = conflict_density(total, conflict_count)
    penalty = coherence_penalty(conflict_count, total - conflict_count)

    plt.scatter(penalty, dens, c="crimson", s=200)
    plt.xlabel("Coherence Penalty")
    plt.ylabel("Conflict Density")
    plt.title("Tessrax AI Memory Contradiction Field")
    plt.grid(True)
    plt.text(penalty + 0.02, dens, f"{conflict_count} conflicts", fontsize=8)
    plt.show()

if __name__ == "__main__":
    plot_memory_contradictions()