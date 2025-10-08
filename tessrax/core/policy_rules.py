"""
Tessrax Policy Rules v1.0
Defines static enforcement patterns for the Governance Kernel.

Each rule has:
  - pattern: regex to match filenames
  - required_tags: list of tags that must appear
  - enforcement: 'warn', 'reject', or 'quorum'
"""

POLICY_RULES = {
    "fork_tag_required": {
        "pattern": r"fork_.*\.py",
        "required_tags": ["fork", "governance"],
        "enforcement": "reject"
    },
    "key_change_requires_quorum": {
        "pattern": r".*key.*\.py",
        "required_tags": ["security"],
        "enforcement": "quorum"
    },
    "scaffolding_must_tag_governance": {
        "pattern": r"scaffolding_engine\.py",
        "required_tags": ["scaffolding", "governance"],
        "enforcement": "warn"
    }
}