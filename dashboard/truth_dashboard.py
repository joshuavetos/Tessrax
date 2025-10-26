"""Streamlit dashboard for the Tessrax Truth API."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import httpx
import pandas as pd
import streamlit as st

API_BASE = st.secrets.get("truth_api_base", "http://localhost:8000")


def fetch_metrics() -> Dict[str, Any]:
    response = httpx.get(f"{API_BASE}/metrics", timeout=5.0)
    response.raise_for_status()
    metrics_data = {}
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, value = line.split()
        metrics_data[name] = float(value)
    return metrics_data


def fetch_self_test() -> Dict[str, Any]:
    response = httpx.get(f"{API_BASE}/self_test", timeout=5.0)
    response.raise_for_status()
    return response.json()


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
