"""Tests validating ledger schema normalization invariants."""
from tessrax.core.ledger.normalize_schema import CANON_FIELDS, normalize_entry


def test_normalization_fields():
    sample = {"event": "test", "payload": {"x": 1}}
    norm = normalize_entry(sample)
    assert list(norm.keys()) == CANON_FIELDS
