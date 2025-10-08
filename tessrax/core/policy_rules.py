"""
Tessrax Policy Rules v1.0
Defines executable governance standards for the system.
"""

POLICY_RULES = {
    "fork_tag_required": {
        "pattern": r"fork_.*\.py",
        "required_tags": ["fork", "governance"],
        "enforcement": "reject"  # Hard block
    },
    "key_change_requires_quorum": {
        "pattern": r".*key.*\.py",
        "required_tags": ["security"],
        "enforcement": "quorum"  # Requires democratic review
    },
    "scaffolding_must_tag_governance": {
        "pattern": r"scaffolding_engine\.py",
        "required_tags": ["scaffolding", "governance"],
        "enforcement": "warn"  # Soft nudge
    }
}