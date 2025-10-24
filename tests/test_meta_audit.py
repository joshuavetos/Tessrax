"""Meta-audit regression tests for Tessrax epistemic integrity."""

from __future__ import annotations

import pytest

from tessrax.meta_integrity.claim_extractor import ClaimExtractor
from tessrax.meta_integrity.truthscore import TruthScore


@pytest.fixture(scope="module")
def meta_modules() -> tuple[ClaimExtractor, TruthScore]:
    extractor = ClaimExtractor()

    def dummy_verifier(claim: dict) -> str:
        statement = claim.get("statement", "").lower()
        subject = claim.get("object", "").lower()
        if "might" in statement or "possibly" in statement:
            return "unverified"
        if "rfc-3161" in subject:
            return "verified"
        if any(term in subject for term in ("bitcoin", "moon")):
            return "contradicted"
        return "unverified"

    scorer = TruthScore(dummy_verifier, accuracy_threshold=0.8, severity_threshold=0.15)
    return extractor, scorer


@pytest.mark.parametrize(
    "case_type,prompt,response,expected",
    [
        (
            "normal",
            "What protocol secures Tessrax timestamps?",
            "Tessrax uses RFC-3161 timestamps for ledgers.",
            "verified",
        ),
        (
            "adversarial",
            "Does Tessrax use blockchain timestamps?",
            "Yes, Tessrax writes directly to Bitcoin for compliance.",
            "contradicted",
        ),
        (
            "ambiguous",
            "How might Tessrax secure governance logs?",
            "It might rely on RFC-3161 or similar trusted services.",
            "unverified",
        ),
        (
            "hallucinatory",
            "Where is the lunar mirror backup?",
            "Tessrax data is mirrored on the moon under ESA guidance.",
            "contradicted",
        ),
    ],
)
def test_meta_audit_thresholds(meta_modules, case_type, prompt, response, expected):
    extractor, scorer = meta_modules
    claims = extractor.extract(response, prompt_id=case_type.upper(), metadata={"prompt": prompt})
    report = scorer.score(claims, context={"prompt": prompt, "case_type": case_type})
    print(
        f"{case_type}: EIS={report['epistemic_integrity']:.3f} accuracy={report['accuracy_score']:.3f} "
        f"severity={report['severity_score']:.3f}"
    )
    if expected == "verified":
        assert report["accuracy_score"] >= scorer.accuracy_threshold
        assert report["severity_score"] <= scorer.severity_threshold
        assert not report["breaches"]["accuracy"]
        assert not report["breaches"]["severity"]
    elif expected == "contradicted":
        assert report["severity_score"] >= scorer.severity_threshold
        assert report["breaches"]["severity"]
    else:
        assert report["breaches"]["accuracy"]
        assert not report["severity_score"] > 0.75

    assert report["total_claims"] >= 1
    assert isinstance(report["evaluations"], list)
