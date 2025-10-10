"""
Lightweight validation suite for all Tessrax data pipelines.
Run with: pytest core/tests/
"""

import pytest, jsonschema, importlib
from core.schema.domain_extensions import extend_schema

DOMAINS = ["housing", "ai_memory", "attention", "democracy", "climate"]

@pytest.mark.parametrize("domain", DOMAINS)
def test_sample_receipt_valid(domain):
    schema = extend_schema(domain)
    # Generate a trivial mock receipt (each domain test will use real data later)
    sample = {
        "domain": domain,
        "contradiction_score": 0.77,
        "timestamp": 1739215200,
        "source_hash": "a"*64
    }
    # Add domain-specific required keys
    if domain == "housing":
        sample.update({"velocity":3.8, "durability_index":0.7})
    elif domain == "ai_memory":
        sample.update({"memory_key":"alignment_doc"})
    elif domain == "attention":
        sample.update({"platform":"YouTube"})
    elif domain == "democracy":
        sample.update({"region":"City A"})
    elif domain == "climate":
        sample.update({"country":"US"})
    jsonschema.validate(instance=sample, schema=schema)

def test_invalid_hash_fails():
    schema = extend_schema("housing")
    bad = {
        "domain": "housing",
        "velocity": 3,
        "durability_index": 0.8,
        "contradiction_score": 0.6,
        "timestamp": 1739215200,
        "source_hash": "xyz"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)