from __future__ import annotations

import pathlib

import pytest

from tessrax.data.evidence_loader import load_field_evidence


def test_load_field_evidence_returns_records() -> None:
    records = load_field_evidence()
    assert isinstance(records, list)
    assert records, "Expected at least one record"
    assert all(isinstance(record, dict) for record in records)


def test_field_evidence_expected_keywords() -> None:
    records = load_field_evidence()
    text_blob = " ".join(
        [
            " ".join(record.get("key_findings", [])) + " " + record.get("summary", "")
            for record in records
        ]
    ).lower()
    for keyword in ("traceability", "audit", "contradiction"):
        assert keyword in text_blob


def test_field_evidence_count() -> None:
    records = load_field_evidence()
    assert len(records) == 22


@pytest.mark.parametrize("path_type", (str, pathlib.Path))
def test_load_field_evidence_supports_path_override(tmp_path: pathlib.Path, path_type: type) -> None:
    source = tmp_path / "sample.jsonl"
    source.write_text(
        '{"id": "test", "category": "demo", "year": 2025, "source_type": "note", '
        '"summary": "demo", "key_findings": ["a"], "alignment": {"policy_reference": "X", "score": 1.0}, '
        '"citations": ["c"]}\n'
    )
    records = load_field_evidence(path_type(source))
    assert records[0]["id"] == "test"
