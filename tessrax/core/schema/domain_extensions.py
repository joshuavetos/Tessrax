"""
Domain-specific schema extensions layered on top of the base contradiction schema.
"""

from copy import deepcopy
from core.schema.contradiction_schema import CONTRADICTION_SCHEMA

def extend_schema(domain:str)->dict:
    s = deepcopy(CONTRADICTION_SCHEMA)
    if domain == "housing":
        s["required"] += ["velocity", "durability_index"]
        s["properties"].update({
            "velocity": {"type": "number"},
            "durability_index": {"type": "number"}
        })
    elif domain == "ai_memory":
        s["required"] += ["memory_key"]
        s["properties"].update({
            "memory_key": {"type": "string"},
            "old_value": {"type": "string"},
            "new_value": {"type": "string"}
        })
    elif domain == "attention":
        s["required"] += ["platform"]
        s["properties"].update({
            "platform": {"type": "string"},
            "session_min": {"type": "number"},
            "wellbeing_score": {"type": "number"}
        })
    elif domain == "democracy":
        s["required"] += ["region"]
        s["properties"].update({
            "region": {"type": "string"},
            "voter_turnout": {"type": "number"},
            "decision_time_days": {"type": "number"}
        })
    elif domain == "climate":
        s["required"] += ["country"]
        s["properties"].update({
            "country": {"type": "string"},
            "gdp_growth": {"type": "number"},
            "emission_change": {"type": "number"}
        })
    return s