"""Tests for the federated ledger visualiser."""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.federation_map import load_federation_state, render_federation_map


def test_load_and_render_federation_map(tmp_path: Path) -> None:
    state_path = tmp_path / "federation_state.jsonl"
    entries = [
        {"node_id": "alpha", "consensus": 0.98, "integrity": 0.96, "label": "Alpha"},
        {"node_id": "beta", "consensus": 0.95, "integrity": 0.94, "label": "Beta"},
    ]
    with state_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    data = load_federation_state(state_path)
    assert len(data) == 2
    html_doc = render_federation_map(data, return_html=True)
    assert "Alpha" in html_doc and "Beta" in html_doc
