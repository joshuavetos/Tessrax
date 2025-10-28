"""Utilities for loading field evidence archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from copy import deepcopy

_DATA_DIR = Path(__file__).resolve().parent / "evidence"
_DEFAULT_DATASET = _DATA_DIR / "field_evidence_archive_2025-10-28.jsonl"

_REQUIRED_FIELDS = {
    "id": str,
    "category": str,
    "year": int,
    "source_type": str,
    "summary": str,
    "key_findings": list,
    "alignment": (dict, Mapping),
    "citations": list,
}

_CACHE: dict[Path, List[Dict[str, Any]]] = {}


def _validate_alignment(alignment: Mapping[str, Any], *, index: int) -> None:
    if not isinstance(alignment, Mapping):
        raise ValueError(f"Entry {index} has invalid alignment payload: {type(alignment)!r}")
    if "policy_reference" not in alignment:
        raise ValueError(f"Entry {index} missing alignment.policy_reference")
    score = alignment.get("score")
    if score is not None and not isinstance(score, (int, float)):
        raise ValueError(f"Entry {index} has non-numeric alignment.score")


def _validate_list(values: Iterable[Any], *, index: int, field: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"Entry {index} expected list for {field}")
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"Entry {index} expected strings in {field}")


def _validate_entry(entry: Dict[str, Any], *, index: int) -> None:
    for field, expected in _REQUIRED_FIELDS.items():
        if field not in entry:
            raise ValueError(f"Entry {index} missing required field '{field}'")
        value = entry[field]
        if field in {"key_findings", "citations"}:
            _validate_list(value, index=index, field=field)
            continue
        if field == "alignment":
            _validate_alignment(value, index=index)
            continue
        expected_types = expected if isinstance(expected, tuple) else (expected,)
        if not isinstance(value, expected_types):
            raise ValueError(
                f"Entry {index} field '{field}' expected {expected_types} but received {type(value)!r}"
            )


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    records: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {index} is not a JSON object")
            _validate_entry(entry, index=index)
            records.append(entry)
    return records


def load_field_evidence(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load and cache field evidence records as structured dictionaries."""

    dataset_path = Path(path) if path is not None else _DEFAULT_DATASET
    dataset_path = dataset_path.resolve()
    if dataset_path not in _CACHE:
        _CACHE[dataset_path] = _load_dataset(dataset_path)
    return deepcopy(_CACHE[dataset_path])


__all__ = ["load_field_evidence"]
