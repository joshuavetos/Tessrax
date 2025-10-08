"""
Tessrax Policy Rules v1.0
Centralized configuration for policy enforcement.

Each rule has:
  - pattern: regex to match filenames
  - required_tags: list of tags that must appear
  - enforcement: 'warn', 'reject', or 'quorum'
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
        "enforcement": "quorum"  # Requires vote
    },
    "scaffolding_must_tag_governance": {
        "pattern": r"scaffolding_engine\.py",
        "required_tags": ["scaffolding", "governance"],
        "enforcement": "warn"    # Soft nudge
    }
}