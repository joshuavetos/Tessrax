"""
housing_contradiction_graph.py
Tessrax v0.1 — Visualize housing contradictions as a scatter plot.
"""

import matplotlib.pyplot as plt
from housing_contradiction_detector import detect_contradictions

def plot_contradictions():
    data = detect_contradictions()
    velocities = [d["velocity"] for d in data]
    durabilities = [d["durability_index"] for d in data]
    scores = [d["contradiction_score"] for d in data]
    labels = [d["location"] for d in data]

    plt.scatter(velocities, durabilities, c=scores, cmap="coolwarm", s=200, alpha=0.8)
    for i, label in enumerate(labels):
        plt.text(velocities[i] + 0.05, durabilities[i], label, fontsize=8)

    plt.xlabel("Transaction Velocity (refinances per decade)")
    plt.ylabel("Durability Index (0–1)")
    plt.title("Tessrax Housing Contradiction Graph")
    plt.colorbar(label="Contradiction Score (0–1)")
    plt.show()

if __name__ == "__main__":
    plot_contradictions()