"""
Agent Validation Harness
Developer-facing test harness for agent responses and contradiction detection.
"""

from tessrax.core.engine_core import analyze


def run_validation(agent_responses):
    """Validate multiple agent claims via Tessrax pipeline."""
    result = analyze(agent_responses)
    print(f"Stability: {result['stability']}, Lane: {result['lane']}")
    return result


if __name__ == "__main__":
    sample_claims = [
        {"agent": "GPT", "claim": "A"},
        {"agent": "Claude", "claim": "A"},
        {"agent": "Gemini", "claim": "B"},
    ]
    run_validation(sample_claims)
