"""Streamlit app visualising live contradiction counts."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

DEFAULT_API_ROOT = os.getenv("TESSRAX_CORE_API", "http://localhost:8000")


@st.experimental_singleton(show_spinner=False)
def _session() -> requests.Session:
    """Provide a shared :class:`requests.Session` for API polling."""

    session = requests.Session()
    session.headers.update({"User-Agent": "Tessrax-Contradiction-Stream/1.0"})
    return session


def _fetch_live(endpoint: str) -> tuple[dict[str, int], str | None]:
    session = _session()
    try:
        response = session.get(endpoint, timeout=3)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {}, str(exc)

    try:
        payload = response.json()
    except ValueError as exc:
        return {}, f"Invalid JSON payload: {exc}"

    if not isinstance(payload, dict):
        return {}, "Response must be a JSON object"
    return payload, None


def _active_series(history: list[dict]) -> list[int]:
    return [int(entry.get("active", 0)) for entry in history]


def main() -> None:
    st.set_page_config(page_title="Tessrax Contradiction Stream", layout="wide")
    st.title("Async Contradiction Stream")

    endpoint_default = f"{DEFAULT_API_ROOT.rstrip('/')}/contradictions/live"
    endpoint = st.text_input("API endpoint", value=endpoint_default)
    st.caption("Polling Tessrax-Core for live contradiction counts.")

    auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
    interval = st.sidebar.slider("Refresh interval (seconds)", 1, 30, value=5)
    if st.sidebar.button("Refresh now"):
        st.experimental_rerun()

    placeholder = st.empty()
    history_key = "contradiction_history"
    history = st.session_state.get(history_key, [])

    payload, error = _fetch_live(endpoint)
    timestamp = time.time()

    if payload:
        history.append({"timestamp": timestamp, **payload})
        st.session_state[history_key] = history[-200:]
        active = int(payload.get("active", 0))
        placeholder.metric("Active Contradictions", active)
        st.line_chart(
            {"Contradictions": _active_series(history)}, use_container_width=True
        )
    else:
        placeholder.warning("No data received yet.")

    if error:
        st.error(f"Live contradiction endpoint unavailable: {error}")

    if auto_refresh:
        time.sleep(interval)
        st.experimental_rerun()


if __name__ == "__main__":  # pragma: no cover - Streamlit entry point
    main()
