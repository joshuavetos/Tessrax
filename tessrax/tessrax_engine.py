"""Core stability and governance routing utilities."""

from __future__ import annotations

from config_loader import load_config

_config = load_config()


def _normalise_claim_text(claim: dict[str, object]) -> str:
    text = ""
    if isinstance(claim, dict):
        for key in ("claim", "content", "text"):
            value = claim.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
    return text


def calculate_stability(claims: list[dict]) -> float:
    """Calculate a stability score using consensus logic."""

    if not claims:
        return 1.0

    texts = [_normalise_claim_text(claim) for claim in claims]
    texts = [text for text in texts if text]
    if not texts:
        return 1.0

    unique_claims = len(set(texts))
    total_claims = len(texts)
    if total_claims == 1:
        return 1.0

    stability = 1.0 - (unique_claims - 1) / (total_claims - 1)
    return max(0.0, min(1.0, round(stability, 6)))


def route_to_governance_lane(
    stability_score: float, thresholds: dict[str, float]
) -> str:
    """Route the request to the appropriate governance lane."""

    defaults = _config.thresholds
    autonomic = thresholds.get("autonomic", defaults.get("autonomic", 0.8))
    deliberative = thresholds.get("deliberative", defaults.get("deliberative", 0.5))
    constitutional = thresholds.get(
        "constitutional", defaults.get("constitutional", 0.3)
    )

    if stability_score >= autonomic:
        return "autonomic"
    if stability_score >= deliberative:
        return "deliberative"
    if stability_score >= constitutional:
        return "constitutional"
    return "behavioral_audit"


__all__ = ["calculate_stability", "route_to_governance_lane"]
