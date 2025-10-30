from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tessrax.metrics import epistemic_health

SCHEMA_PATH = Path("schemas/epistemic_metrics.schema.json")
VECTORS_DIR = Path("tests/data/epistemic_metrics")


@pytest.fixture(scope="session")
def schema() -> Draft202012Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    return Draft202012Validator(document)


@pytest.mark.parametrize("vector_path", sorted(VECTORS_DIR.glob("*.json")))
def test_vector_compliance(schema: Draft202012Validator, vector_path: Path) -> None:
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    schema.validate(payload)

    metrics = payload["metrics"]
    integrity = epistemic_health.compute_integrity([0.2, 0.3, 0.4, 0.5])
    drift = epistemic_health.compute_drift([(0.0, 0.3), (1.0, 0.4), (2.0, 0.42)])
    severity = epistemic_health.compute_severity([0.1, 0.2, 0.4], [0.1, 0.25, 0.33])
    independence = epistemic_health.compute_entropy(["a", "b", "b", "c", "c", "c"])

    assert pytest.approx(metrics["integrity"], abs=1e-4) == integrity
    assert pytest.approx(metrics["drift"], abs=1e-4) == drift
    assert pytest.approx(metrics["severity"], abs=1e-4) == severity
    assert pytest.approx(metrics["independence"], abs=1e-4) == independence
