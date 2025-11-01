from __future__ import annotations

from datetime import datetime, timezone

from tessrax.governance import GovernanceKernel
from tessrax.governance_security import DecisionSignature, SignatureAuthority
from tessrax.types import Claim, ContradictionRecord


def _claim(claim_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        subject="biosphere",
        metric="oxygen",
        value=21.0,
        unit="%",
        timestamp=datetime.now(timezone.utc),
        source="sensor",
        context={},
    )


def test_governance_kernel_attaches_signature() -> None:
    authority = SignatureAuthority(secret="governance-secret")
    kernel = GovernanceKernel(signature_authority=authority)
    record = ContradictionRecord(
        claim_a=_claim("claim-a"),
        claim_b=_claim("claim-b"),
        severity="medium",
        delta=0.12,
        reasoning="Discrepancy in redundant telemetry.",
    )

    decision = kernel.process(record)
    assert decision.signature is not None
    assert decision.timestamp_token is not None

    signature = DecisionSignature(
        signature=decision.signature, timestamp_token=decision.timestamp_token
    )
    assert authority.verify(decision, signature)


def test_signature_verification_detects_tampering() -> None:
    authority = SignatureAuthority(secret="governance-secret")
    kernel = GovernanceKernel(signature_authority=authority)
    record = ContradictionRecord(
        claim_a=_claim("claim-a"),
        claim_b=_claim("claim-b"),
        severity="high",
        delta=0.45,
        reasoning="Major conflict in telemetry streams.",
    )

    decision = kernel.process(record)
    signature = DecisionSignature(
        signature=decision.signature, timestamp_token=decision.timestamp_token
    )

    # Tamper with the decision payload and ensure verification fails.
    decision.rationale = "Tampered rationale"
    assert authority.verify(decision, signature) is False
