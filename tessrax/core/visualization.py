#!/usr/bin/env python3
"""
Tessrax Visualization Engine — Unified Graph + Drift + Heatmap Viewer
---------------------------------------------------------------------
Merged modules:
  • visualize_scaffolding.py
  • plot_stability_drift.py
  • agent_heatmap.py

Purpose:
  Provide all visual analytics for the Tessrax Core system.
  Includes:
  - Contradiction Graph Visualization
  - Ledger Stability Drift Plot
  - Agent Agreement Heatmap
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tessrax.core.engine import detect_contradictions
from tessrax.core.ledger import load_all_entries


# ----------------------------------------------------------------------
# Contradiction Graph Visualization
# ----------------------------------------------------------------------

def visualize_contradiction_graph(agent_claims: List[Dict[str, Any]], title: str = "Contradiction Graph") -> None:
    """
    Render contradiction relationships between agents.
    Red edges = contradictions, gray = agreement.
    """
    G = detect_contradictions(agent_claims)
    pos = nx.spring_layout(G, seed=42)
    edge_colors = ["red" if d.get("contradiction") else "gray" for _, _, d in G.edges(data=True)]

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        edge_color=edge_colors,
        node_color="#B0E0E6",
        node_size=1500,
        font_size=10,
        font_weight="bold",
        width=2
    )
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Ledger Stability Drift Plot
# ----------------------------------------------------------------------

def plot_stability_drift(ledger_path: Path = Path("data/ledger.jsonl")) -> None:
    """
    Plot stability index drift over time based on ledger entries.
    """
    if not ledger_path.exists():
        print(f"[Error] Ledger file not found: {ledger_path}")
        return

    timestamps, stability = [], []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                timestamps.append(entry.get("timestamp"))
                stability.append(entry.get("stability_index"))
            except Exception:
                continue

    if not stability:
        print("[Warning] No valid stability data found in ledger.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(stability)), stability, marker="o", linestyle="-", color="deepskyblue")
    plt.title("Stability Index Drift Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Event Index")
    plt.ylabel("Stability Index")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Agent Agreement Heatmap
# ----------------------------------------------------------------------

def plot_agent_agreement(agent_claims: List[Dict[str, Any]], title: str = "Agent Agreement Heatmap") -> None:
    """
    Render an N×N heatmap showing agreement vs. contradiction frequency.
    """
    agents = [c["agent"] for c in agent_claims]
    n = len(agents)
    matrix = np.zeros((n, n))

    # Build contradiction matrix
    for i in range(n):
        for j in range(n):
            matrix[i, j] = 1 if agent_claims[i]["claim"] != agent_claims[j]["claim"] else 0

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="1 = Contradiction, 0 = Agreement")
    plt.xticks(range(n), agents, rotation=90)
    plt.yticks(range(n), agents)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Ledger Overview Dashboard
# ----------------------------------------------------------------------

def render_ledger_dashboard() -> None:
    """
    Render an integrated dashboard combining stability drift and agent consensus trends.
    """
    entries = load_all_entries()
    if not entries:
        print("[Warning] No ledger data found.")
        return

    # Extract metrics
    stability = [e.get("stability_index", 0) for e in entries if "stability_index" in e]
    lanes = [e.get("governance_lane", e.get("lane", "Unknown")) for e in entries]
    lane_labels = list(set(lanes))
    lane_counts = [lanes.count(l) for l in lane_labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Stability drift
    axes[0].plot(stability, marker="o", color="royalblue")
    axes[0].set_title("Ledger Stability Over Time", fontsize=12)
    axes[0].set_xlabel("Entry Index")
    axes[0].set_ylabel("Stability Index")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Lane distribution
    axes[1].bar(lane_labels, lane_counts, color="orange")
    axes[1].set_title("Governance Lane Distribution", fontsize=12)
    axes[1].set_xlabel("Lane")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Tessrax Governance Ledger Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Demo Entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Tessrax Visualization Demo ---")

    # Demo dataset
    demo_claims = [
        {"agent": "GPT", "claim": "A", "type": "normative"},
        {"agent": "Claude", "claim": "B", "type": "normative"},
        {"agent": "Gemini", "claim": "A", "type": "normative"},
        {"agent": "Grok", "claim": "B", "type": "normative"},
    ]

    print("[1] Contradiction Graph")
    visualize_contradiction_graph(demo_claims)

    print("[2] Agent Agreement Heatmap")
    plot_agent_agreement(demo_claims)

    print("[3] Ledger Stability Drift (if ledger exists)")
    plot_stability_drift()

    print("[4] Ledger Dashboard Overview")
    render_ledger_dashboard()
