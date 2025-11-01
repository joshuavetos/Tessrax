"""Policy compiler converting Iron Laws to executable validators.

Complies with AEP-001, RVC-001, and POST-AUDIT-001 while generating
deterministic validators for Tessrax governance claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tessrax.core.governance.receipts import write_receipt


@dataclass(frozen=True)
class CompiledLaw:
    """Lightweight container storing parsed Iron Law instructions."""

    keyword: str
    truth_status: str
    confidence: float


def compile_iron_laws(laws: Sequence[str]) -> list[CompiledLaw]:
    """Compile plain-text Iron Laws into deterministic representations."""

    compiled: list[CompiledLaw] = []
    for law in laws:
        body = law.replace("Iron Law:", "").strip()
        keyword_part, remainder = body.split("->")
        keyword = keyword_part.strip().lower()
        status_part, confidence_part = remainder.split("@")
        truth_status = status_part.strip().lower()
        confidence = float(confidence_part.strip())
        compiled.append(
            CompiledLaw(
                keyword=keyword, truth_status=truth_status, confidence=confidence
            )
        )
    return compiled


def evaluate_claim(
    claim: dict[str, str | float], laws: Sequence[CompiledLaw]
) -> list[dict[str, str | float | bool]]:
    """Evaluate a claim against compiled Iron Laws."""

    text = claim.get("text", "").lower()
    claim_confidence = float(claim.get("confidence", 0.0))
    evaluations: list[dict[str, str | float | bool]] = []
    for law in laws:
        keyword_present = law.keyword in text
        verdict_confidence = min(claim_confidence, law.confidence)
        evaluations.append(
            {
                "keyword_present": keyword_present,
                "truth_status": law.truth_status if keyword_present else "unverified",
                "confidence": round(verdict_confidence, 6),
                "keyword": law.keyword,
            }
        )
    write_receipt(
        "tessrax.core.governance.policy_compiler",
        "verified",
        {"evaluations": len(evaluations)},
        0.95,
    )
    return evaluations


def _self_test() -> bool:
    """Run deterministic compilation and evaluation checks."""

    laws = [
        "Iron Law: solar output -> verified @0.9",
        "Iron Law: anomaly -> review @0.6",
    ]
    compiled = compile_iron_laws(laws)
    claim = {"text": "Solar output anomaly detected", "confidence": 0.88}
    evaluations = evaluate_claim(claim, compiled)
    statuses = [item["truth_status"] for item in evaluations]
    assert statuses[0] == "verified", "Expected verified status"
    assert evaluations[1]["truth_status"] == "review", "Expected review status"
    assert evaluations[0]["confidence"] == 0.88, "Confidence mismatch"
    write_receipt(
        "tessrax.core.governance.policy_compiler.self_test",
        "verified",
        {"evaluations": len(evaluations)},
        0.95,
    )
    return True


if __name__ == "__main__":
    assert _self_test(), "Self-test failed"
