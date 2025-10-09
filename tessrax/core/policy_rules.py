"""
Tessrax Policy Rules v1.1
Defines executable governance standards for the Tessrax Stack.
"""

import re
from datetime import datetime

# ============================================================
# Canonical Policy Registry
# ============================================================

POLICY_RULES = {
    "fork_tag_required": {
        "pattern": r"fork_.*\.py",
        "required_tags": ["fork", "governance"],
        "enforcement": "reject",  # Hard block
        "description": "Fork-related modules must include 'fork' and 'governance' tags.",
    },

    "key_change_requires_quorum": {
        "pattern": r".*key.*\.py",
        "required_tags": ["security"],
        "enforcement": "quorum",  # Requires democratic review
        "description": "Any module touching key material requires quorum validation.",
    },

    "scaffolding_must_tag_governance": {
        "pattern": r"scaffolding_engine\.py",
        "required_tags": ["scaffolding", "governance"],
        "enforcement": "warn",  # Soft nudge
        "description": "Scaffolding modules must explicitly tag governance and scaffolding.",
    },
}

# ============================================================
# Utility Functions
# ============================================================

def list_policies():
    """Return all available policy names and enforcement levels."""
    return {k: v["enforcement"] for k, v in POLICY_RULES.items()}

def validate_file_against_policies(filename: str, tags: list[str]) -> list[dict]:
    """
    Evaluate a file + tag set against policy rules.
    Returns list of violations with details.
    """
    results = []
    tagset = set(tags)

    for name, rule in POLICY_RULES.items():
        if re.match(rule["pattern"], filename):
            missing = [t for t in rule["required_tags"] if t not in tagset]
            if missing:
                results.append({
                    "policy": name,
                    "missing_tags": missing,
                    "enforcement": rule["enforcement"],
                    "timestamp": datetime.utcnow().isoformat(),
                })

    return results


if __name__ == "__main__":
    # Quick smoke test for developer use
    print("🧩 Tessrax Policy Rules – Diagnostic Mode")
    test = validate_file_against_policies("fork_reconciliation_engine.py", ["core"])
    print(test or "✓ No violations detected.")