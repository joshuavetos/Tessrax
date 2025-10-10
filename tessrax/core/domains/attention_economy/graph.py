"""
attention_contradiction_graph.py
Visualizes engagement vs. well-being contradictions.
"""

import matplotlib.pyplot as plt
from attention_contradiction_detector import detect_attention_conflicts

def plot_attention_field():
    data = detect_attention_conflicts()
    x = [d["session_min"] for d in data]
    y = [d["wellbeing_score"] for d in data]
    c = [d["contradiction_score"] for d in data]
    labels = [d["platform"] for d in data]

    plt.scatter(x, y, c=c, cmap="coolwarm", s=220, alpha=0.8)
    for i, lbl in enumerate(labels):
        plt.text(x[i] + 2, y[i], lbl, fontsize=8)
    plt.xlabel("Average Session Length (minutes)")
    plt.ylabel("Well-being Score (0–1)")
    plt.title("Tessrax Attention Contradiction Map")
    plt.colorbar(label="Contradiction Score (0–1)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_attention_field()