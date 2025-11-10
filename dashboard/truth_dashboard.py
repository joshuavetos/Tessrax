"""Streamlit dashboard for the Tessrax Truth API."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE = st.secrets.get("truth_api_base", "http://localhost:8000")

FALLBACK_METRICS = {
    "truth_api_integrity": 0.0,
    "truth_api_drift": 0.0,
    "truth_api_severity": 0.0,
}

FALLBACK_SELF_TEST = {
    "results": [
        {
            "name": "offline",
            "status": "unknown",
            "receipt_uuid": None,
            "details": "API offline",
        }
    ],
    "ledger_path": "offline",
}


def fetch_metrics() -> dict[str, Any]:
    try:
        response = httpx.get(f"{API_BASE}/metrics", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.warning(f"Truth API metrics unavailable: {exc}")
        return dict(FALLBACK_METRICS)
    metrics_data: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            name, value = line.split()
            metrics_data[name] = float(value)
        except ValueError:
            continue
    return metrics_data or dict(FALLBACK_METRICS)


def fetch_self_test() -> dict[str, Any]:
    try:
        response = httpx.get(f"{API_BASE}/self_test", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.warning(f"Truth API self-test unavailable: {exc}")
        return dict(FALLBACK_SELF_TEST)
    payload = response.json()
    if "results" not in payload or not isinstance(payload["results"], list):
        st.warning(
            "Truth API self-test returned unexpected payload; using fallback data."
        )
        return dict(FALLBACK_SELF_TEST)
    return payload


st.set_page_config(page_title="Tessrax Truth Dashboard", layout="wide")
st.title("Tessrax Truth Dashboard")

metrics = fetch_metrics()
cols = st.columns(3)
cols[0].metric("Integrity", f"{metrics.get('truth_api_integrity', 0):.2f}")
cols[1].metric("Drift", f"{metrics.get('truth_api_drift', 0):.2f}")
cols[2].metric("Severity", f"{metrics.get('truth_api_severity', 0):.2f}")

self_test = fetch_self_test()
st.subheader("Self Test Results")
results = pd.DataFrame(self_test["results"])
st.dataframe(results)
st.caption(f"Ledger: {self_test['ledger_path']}")

st.subheader("Raw Metrics")
st.code(json.dumps(metrics, indent=2))
