"""Streamlit dashboard for Tessrax-Core ledgers."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit.components.v1 import html

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane

config = load_config()
default_ledger = Path(config.logging.ledger_path)
DEFAULT_EPI_ENDPOINT = os.getenv("TESSRAX_GOVERNANCE_API", "http://localhost:8000/epistemic_metrics")
POLL_INTERVAL_SECONDS = 30
HISTORY_STATE_KEY = "epistemic_metrics_history"


@st.cache_data(show_spinner=False)
def _load_ledger(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _enrich_records(records: Iterable[dict]) -> List[dict]:
    enriched: List[dict] = []
    for record in records:
        claims = record.get("claims", [])
        stability = calculate_stability(claims)
        lane = route_to_governance_lane(stability, config.thresholds)
        enriched.append({**record, "stability_score": stability, "governance_lane": lane})
    return enriched


def _fetch_epistemic_metrics(url: str) -> tuple[dict, str | None]:
    """Pull the latest epistemic metrics from the governance API."""

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {}, f"{exc}"
    except ValueError as exc:
        return {}, f"Invalid JSON payload: {exc}"
    if not isinstance(payload, dict):
        return {}, "Epistemic metrics payload must be a JSON object"
    return payload, None


def _record_metrics_history(entry: dict[str, float]) -> list[dict[str, float]]:
    """Persist metrics history in Streamlit session state."""

    history = st.session_state.get(HISTORY_STATE_KEY, [])
    history.append({"timestamp": datetime.utcnow().isoformat(), **entry})
    st.session_state[HISTORY_STATE_KEY] = history[-120:]
    return st.session_state[HISTORY_STATE_KEY]


def _summarise(records: Iterable[dict]) -> dict:
    lanes = Counter(record.get("governance_lane", "unknown") for record in records)
    return {"total": sum(lanes.values()), "lanes": dict(lanes)}


def render_integrity_gradient(clarity_records: List[dict], alert_threshold: float = 0.05) -> None:
    """Visualise reconciliation integrity with drift overlays and alerts."""

    if not clarity_records:
        st.info("Integrity gradient unavailable – no telemetry yet.")
        return
    frame = pd.DataFrame(clarity_records)
    if {"timestamp", "integrity_score", "domain"} - set(frame.columns):
        st.warning("Integrity gradient requires timestamp, domain, and integrity_score fields.")
        return
    frame = frame.sort_values("timestamp")
    frame["rolling_mean"] = frame.groupby("domain")["integrity_score"].transform(
        lambda series: series.rolling(10, min_periods=1).mean()
    )
    fig = px.line(
        frame,
        x="timestamp",
        y="integrity_score",
        color="domain",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Integrity Gradient — Contradiction → Clarity Over Time",
    )
    fig.add_scatter(
        x=frame["timestamp"],
        y=frame["rolling_mean"],
        mode="lines",
        name="Rolling Avg",
        line=dict(dash="dot"),
    )
    st.plotly_chart(fig, use_container_width=True)

    recent_window = frame.tail(5)
    if not recent_window.empty:
        drift = abs(recent_window["integrity_score"].mean() - recent_window["rolling_mean"].iloc[-1])
        if drift > alert_threshold:
            st.warning(f"⚠️ Epistemic drift anomaly detected ({drift:.3f})")


def main() -> None:
    st.set_page_config(page_title="Tessrax Ledger Dashboard", layout="wide")
    st.title("Tessrax Governance Ledger")

    ledger_records = _enrich_records(_load_ledger(default_ledger))

    uploaded = st.file_uploader("Override ledger", type=["jsonl", "txt"])
    if uploaded:
        uploaded_path = default_ledger.parent / "_uploaded_ledger.jsonl"
        uploaded_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded_path.write_bytes(uploaded.getvalue())
        ledger_records = _enrich_records(_load_ledger(uploaded_path))

    summary = _summarise(ledger_records)

    st.metric("Ledger events", summary["total"])
    st.json(summary["lanes"], expanded=False)

    if ledger_records:
        st.subheader("Latest entry")
        st.json(ledger_records[-1], expanded=False)

        st.subheader("Ledger records")
        st.dataframe(ledger_records)
    else:
        st.info(f"No ledger records found at {default_ledger}")

    st.divider()
    st.subheader("Epistemic Metrics")
    html(f"<script>setTimeout(() => window.location.reload(), {POLL_INTERVAL_SECONDS * 1000});</script>", height=0)
    st.caption(
        f"Polling {DEFAULT_EPI_ENDPOINT} every {POLL_INTERVAL_SECONDS} seconds for epistemic integrity telemetry."
    )
    metrics, error = _fetch_epistemic_metrics(DEFAULT_EPI_ENDPOINT)
    if error:
        st.warning(f"Epistemic metrics unavailable: {error}")
    else:
        history = _record_metrics_history(metrics)
        integrity = float(metrics.get("epistemic_integrity", 0.0))
        drift_values = [entry.get("epistemic_drift", 0.0) for entry in history]
        resilience = float(metrics.get("adversarial_resilience", 0.0))
        hallucination = float(metrics.get("hallucination_rate", 0.0))

        clarity_records = [
            {
                "timestamp": entry["timestamp"],
                "integrity_score": entry.get("epistemic_integrity", 0.0),
                "domain": "global",
            }
            for entry in history
        ]

        col1, col2 = st.columns(2)
        gauge_value = max(0.0, min(integrity, 1.0))
        col1.metric("Epistemic Integrity Score", f"{integrity:.2%}", delta="Target ≥ 85%")
        col1.progress(int(gauge_value * 100))
        col2.metric("Adversarial Resilience", f"{resilience:.2%}", delta="Target ≥ 90%")
        col2.bar_chart({"Resilience": [resilience]}, use_container_width=True)

        st.line_chart({"Epistemic Drift": drift_values}, use_container_width=True)
        hallux = max(0.0, min(hallucination, 1.0))
        heatmap_html = f"""
        <div style="display:flex;flex-direction:column;gap:0.25rem;">
            <div style="font-weight:600;">Hallucination Rate</div>
            <div style="background:rgba(255,99,71,{hallux});padding:0.75rem;border-radius:0.5rem;color:white;">
                {hallucination:.2%} (≤ 10%)
            </div>
        </div>
        """
        st.markdown(heatmap_html, unsafe_allow_html=True)
        if hallux > 0.1:
            st.error("Hallucination rate exceeds target", icon="🚨")
        else:
            st.success("Hallucination rate within target", icon="✅")
        st.caption(f"Metrics updated: {datetime.utcnow().isoformat()}Z")
        st.subheader("Integrity Gradient")
        render_integrity_gradient(clarity_records)


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
