"""
Tessrax Visualization Scaffold (VIS-MOD-03)
--------------------------------------------

Visualization layer for contradiction and governance events.

Provides:
    • Real-time contradiction graph rendering
    • Stability heatmap
    • Governance lane color-coding
    • Optional interactive Matplotlib display

Version: VIS-MOD-03-R2
Author: Tessrax LLC
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx


# === COLOR SCHEME ============================================================

LANE_COLORS = {
    "Autonomic": "#9BE564",
    "Deliberative": "#F4D35E",
    "Constitutional": "#EE964B",
    "Behavioral Audit": "#E63946",
}

EDGE_COLORS = {
    "logical": "#C6DEF1",
    "temporal": "#F7D6BF",
    "semantic": "#E2C2B9",
    "normative": "#C9ADA7",
}


# === CORE GRAPH VISUALIZATION ================================================

def visualize_graph(G: nx.Graph, stability: float, lane: str, title: str = "Tessrax Contradiction Graph") -> None:
    """
    Render a contradiction graph with governance lane context.

    Args:
        G: networkx Graph from CE-MOD-66
        stability: stability score (0–1)
        lane: governance lane string
        title: plot title
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Node styling
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#B8D8BA",
        node_size=1200,
        alpha=0.9,
        linewidths=1.5,
        edgecolors="#5A5A5A"
    )

    # Edge coloring by contradiction type
    edge_colors = [
        EDGE_COLORS.get(d.get("type", "normative"), "#CCCCCC")
        for _, _, d in G.edges(data=True)
    ]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2.5, alpha=0.8)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="#2E2E2E")

    # Title + meta
    plt.title(
        f"{title}\nLane: {lane} | Stability: {stability:.3f}",
        fontsize=14,
        fontweight="bold",
        color=LANE_COLORS.get(lane, "#FFFFFF"),
        pad=20
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# === LEDGER VISUALIZATION =====================================================

def visualize_stability_drift(ledger_path: str = "data/governance_ledger.jsonl") -> None:
    """
    Plot stability drift over time from the governance ledger.
    """
    if not Path(ledger_path).exists():
        print(f"Ledger file not found: {ledger_path}")
        return

    entries = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        print("No valid entries found in ledger.")
        return

    timestamps = [e["timestamp"] for e in entries]
    stability = [e["stability_index"] for e in entries]
    lanes = [e["governance_lane"] for e in entries]

    colors = [LANE_COLORS.get(lane, "#CCCCCC") for lane in lanes]

    plt.figure(figsize=(10, 5))
    plt.scatter(timestamps, stability, c=colors, s=100, edgecolor="k", alpha=0.85)
    plt.plot(stability, color="#666666", linestyle="--", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Stability Index (0–1)")
    plt.title("Governance Stability Drift Over Time", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# === DEMO ENTRY ===============================================================

def demo(agent_claims: List[Dict[str, str]], engine, kernel) -> None:
    """
    Run a full demo: contradiction → governance → visualization.
    """
    G, stability = engine.run_contradiction_cycle(agent_claims)
    event = kernel.route(G, stability)

    print(f"\nGovernance Lane: {event.governance_lane}")
    print(f"Stability Index: {event.stability_index}")
    print(f"Agents: {', '.join(event.agents)}")
    print(f"Note: {event.note}\n")

    visualize_graph(G, stability, event.governance_lane)
    visualize_stability_drift()
