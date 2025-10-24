"""Streamlit dashboard for Tessrax-Core ledgers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import streamlit as st

from config_loader import load_config
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane

config = load_config()
default_ledger = Path(config.logging.ledger_path)


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


def _summarise(records: Iterable[dict]) -> dict:
    lanes = Counter(record.get("governance_lane", "unknown") for record in records)
    return {"total": sum(lanes.values()), "lanes": dict(lanes)}


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


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
