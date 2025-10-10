"""
Defines canonical JSON Schema for Tessrax contradiction receipts.
Used by validators across all domain pipelines.
"""

CONTRADICTION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Tessrax Contradiction Receipt",
    "type": "object",
    "required": ["domain", "contradiction_score", "timestamp", "source_hash"],
    "properties": {
        "domain": {"type": "string"},
        "location": {"type": ["string", "null"]},
        "metadata": {"type": ["object", "null"]},
        "contradiction_score": {"type": "number", "minimum": 0, "maximum": 1},
        "timestamp": {"type": "integer"},
        "source_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"}
    },
    "additionalProperties": True
}