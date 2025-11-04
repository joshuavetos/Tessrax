"""Federated ledger visualiser with accessibility features (DLK-verified).

The visualiser consumes `ledger/federation_state.jsonl`, renders a D3-driven
node map inside Streamlit, and exports a static PNG snapshot for audit trails.
It operates under Tessrax Governance Kernel v16 and enforces the clauses
["AEP-001","POST-AUDIT-001","RVC-001","EAC-001"].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont

LEDGER_STATE_PATH = Path("ledger/federation_state.jsonl")
SNAPSHOT_PATH = Path("dashboard/static/federation_map.png")
OKABE_ITO = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def load_federation_state(path: Path = LEDGER_STATE_PATH) -> list[dict[str, object]]:
    """Load federation metadata from a JSONL ledger snapshot."""

    if not path.exists():
        return []
    data: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _build_d3_payload(nodes: Iterable[dict[str, object]]) -> str:
    mapped = []
    for index, node in enumerate(nodes):
        palette_colour = OKABE_ITO[index % len(OKABE_ITO)]
        mapped.append(
            {
                "id": node.get("node_id", f"node-{index}"),
                "consensus": float(node.get("consensus", 0.0)),
                "integrity": float(node.get("integrity", 0.0)),
                "label": node.get("label", f"Node {index}"),
                "colour": palette_colour,
            }
        )
    return json.dumps(mapped, sort_keys=True)


def render_federation_map(
    nodes: Iterable[dict[str, object]] | None = None,
    *,
    return_html: bool = False,
) -> str | None:
    """Render the federated ledger map using Streamlit and D3."""

    payload = _build_d3_payload(nodes or load_federation_state())
    html_doc = f"""
    <div id="viz" role="img" aria-label="Federation integrity map" tabindex="0"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const data = {payload};
        const width = 700, height = 500;
        const svg = d3.select("#viz").append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("role", "img")
            .attr("aria-label", "Federation integrity map");
        const tooltip = d3.select("body").append("div")
            .attr("role", "tooltip")
            .attr("tabindex", "0")
            .style("position", "absolute")
            .style("padding", "8px")
            .style("background", "#111")
            .style("color", "#fff")
            .style("border-radius", "4px")
            .style("pointer-events", "none")
            .style("opacity", 0);
        const simulation = d3.forceSimulation(data)
            .force("charge", d3.forceManyBody().strength(-120))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => 40 + d.integrity * 20));
        const nodes = svg.selectAll("circle").data(data).enter().append("circle")
            .attr("r", d => 20 + d.integrity * 20)
            .attr("fill", d => d.colour)
            .attr("stroke", "#000")
            .attr("stroke-width", 1.5)
            .attr("tabindex", "0")
            .on("focus mouseover", (event, d) => {{
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(`<strong>${{d.label}}</strong><br/>Consensus: ${{(d.consensus*100).toFixed(1)}}%<br/>Integrity: ${{(d.integrity*100).toFixed(1)}}%`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("blur mouseout", () => tooltip.transition().duration(300).style("opacity", 0));
        simulation.on("tick", () => {{
            nodes.attr("cx", d => d.x).attr("cy", d => d.y);
        }});
    </script>
    """
    if return_html:
        return html_doc

    try:
        import streamlit as st
        from streamlit.components.v1 import html
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError("Streamlit is required to render the federation map") from exc

    st.set_page_config(page_title="Federation Map", page_icon="🛰", layout="wide")
    st.title("Federated Ledger Integrity Map")
    st.caption("Colourblind-safe Okabe–Ito palette with keyboard tooltips")
    html(html_doc, height=520)
    return None


def export_snapshot(nodes: Iterable[dict[str, object]] | None = None, path: Path = SNAPSHOT_PATH) -> Path:
    """Export a simple PNG snapshot summarising federation integrity."""

    entries: List[dict[str, object]] = list(nodes or load_federation_state())
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (800, 400), color="#0f172a")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - fallback
        font = None
    draw.text((20, 20), "Federation Integrity Snapshot", fill="#f8fafc", font=font)
    y = 80
    for index, node in enumerate(entries[:10]):
        label = node.get("label", node.get("node_id", f"Node {index}"))
        consensus = float(node.get("consensus", 0.0)) * 100
        integrity = float(node.get("integrity", 0.0)) * 100
        colour = OKABE_ITO[index % len(OKABE_ITO)]
        draw.rectangle([(20, y - 10), (40, y + 10)], fill=colour)
        draw.text((60, y - 10), f"{label}: C={consensus:.1f}% I={integrity:.1f}%", fill="#e2e8f0", font=font)
        y += 40
    image.save(path)
    return path
