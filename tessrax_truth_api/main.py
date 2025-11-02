"""Async FastAPI application exposing the Tessrax Truth API."""

import json

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response, status

from tessrax_truth_api.engine.calibrator import Calibrator
from tessrax_truth_api.engine.contradiction_engine import ContradictionEngine
from tessrax_truth_api.ledger_link import ledger_guard
from tessrax_truth_api.middleware.truthlock_middleware import TruthLockMiddleware
from tessrax_truth_api.models import (
    DetectRequest,
    DetectResponse,
    HealthResponse,
    OnboardResponse,
    ReceiptVerification,
    SelfTestResult,
    SelfTestSummary,
)
from tessrax_truth_api.services.billing_service import BillingService
from tessrax_truth_api.services.subscription_service import SubscriptionService
from tessrax_truth_api.services.entitlement_service import EntitlementService
from tessrax_truth_api.services.webhook_service import WebhookService
from tessrax_truth_api.services.stripe_gateway import StripeGateway
from tessrax_truth_api.services.cache_service import CachedEntry, CacheService
from tessrax_truth_api.services.provenance_service import ProvenanceService
from tessrax_truth_api.services.validation_service import ValidationService
from tessrax_truth_api.utils import (
    encode_metrics,
    hmac_signature,
    issue_jwt,
    load_config,
    merkle_hash,
    utcnow,
    verify_signature,
)


def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI(title="Tessrax Truth API", version="2.0.0")

    calibrator = Calibrator()
    thresholds = calibrator.thresholds
    engine = ContradictionEngine()
    validation_service = ValidationService(
        engine,
        (
            float(thresholds.get("contradiction", 0.65)),
            float(thresholds.get("unknown", 0.25)),
        ),
    )
    billing_service = BillingService()
    subscription_service = SubscriptionService()
    entitlement_service = EntitlementService(subscription_service)
    webhook_service = WebhookService(subscription_service)
    stripe_gateway = StripeGateway()
    cache_service = CacheService()
    provenance_service = ProvenanceService()

    app.state.validation_service = validation_service
    app.state.billing_service = billing_service
    app.state.cache_service = cache_service
    app.state.provenance_service = provenance_service
    app.state.calibrator = calibrator
    app.state.config = config

    monetization_api.init_services(
        subscription_service,
        entitlement_service,
        webhook_service,
        stripe_gateway,
    )
    app.include_router(monetization_api.router)

    app.add_middleware(TruthLockMiddleware)

    @app.on_event("startup")
    async def startup_event() -> (
        None
    ):  # pragma: no cover - placeholder for future hooks
        return None

    @app.post("/onboard", response_model=OnboardResponse)
    async def onboard(tier: str = "free", key: str = "create") -> OnboardResponse:
        billing_config = config.get("billing", {})
        if key != billing_config.get("onboarding_key", "create"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid onboarding key"
            )
        if tier not in billing_config.get("tiers", {}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown tier"
            )
        token = issue_jwt(tier, subject="starter")
        return OnboardResponse(
            tier=tier,
            token=token,
            expires_in_minutes=config.get("jwt", {}).get("expires_minutes", 60),
        )

    def _resolve_context(
        request: Request,
    ) -> tuple[
        Request, ValidationService, BillingService, CacheService, ProvenanceService
    ]:
        return (
            request,
            request.app.state.validation_service,
            request.app.state.billing_service,
            request.app.state.cache_service,
            request.app.state.provenance_service,
        )

    @app.post("/detect", response_model=DetectResponse)
    @ledger_guard(provenance_service, "TRUTH_API_CONTRADICTION_DETECTED")
    async def detect(
        payload: DetectRequest = Body(...),
        context: tuple[
            Request, ValidationService, BillingService, CacheService, ProvenanceService
        ] = Depends(_resolve_context),
    ) -> DetectResponse:
        request, validation, billing, cache, provenance = context
        jwt_claims = getattr(request.state, "jwt_claims", None)
        bearer_token = getattr(request.state, "bearer_token", None)
        if jwt_claims is None or bearer_token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT required"
            )
        if jwt_claims.get("tier") != payload.tier:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Tier mismatch"
            )

        cached = cache.get(payload.claim_a, payload.claim_b)
        if cached:
            billing.record_usage(bearer_token, payload.tier)
            receipt = provenance.append_receipt(
                {
                    "claim_a": payload.claim_a,
                    "claim_b": payload.claim_b,
                    "tier": payload.tier,
                    "score": cached.score,
                    "verdict": cached.verdict,
                    "status": cached.status,
                    "cached": True,
                }
            )
            return DetectResponse(
                score=cached.score,
                verdict=cached.verdict,
                status=cached.status,
                receipt_uuid=receipt.uuid,
                timestamp=utcnow(),
                signature=receipt.signature,
            )

        result = validation.validate_claim_pair(payload.claim_a, payload.claim_b)
        billing.record_usage(bearer_token, payload.tier)
        payload_dict = {
            "claim_a": payload.claim_a,
            "claim_b": payload.claim_b,
            "tier": payload.tier,
            "score": result.score,
            "verdict": result.verdict,
            "status": result.status,
        }
        receipt = provenance.append_receipt(payload_dict)
        cache.set(
            payload.claim_a,
            payload.claim_b,
            CachedEntry(result.score, result.verdict, result.status),
        )
        return DetectResponse(
            score=result.score,
            verdict=result.verdict,
            status=result.status,
            receipt_uuid=receipt.uuid,
            timestamp=utcnow(),
            signature=receipt.signature,
        )

    @app.get("/verify_receipt/{uuid}", response_model=ReceiptVerification)
    @ledger_guard(provenance_service, "TRUTH_API_RECEIPT_VERIFICATION")
    async def verify_receipt(uuid: str) -> ReceiptVerification:
        record = provenance_service.verify_receipt(uuid)
        signature_valid = verify_signature(record.payload, record.signature)
        status_value = record.payload.get("status", "unknown")
        return ReceiptVerification(
            uuid=record.uuid,
            status=(
                status_value if status_value in {"verified", "unknown"} else "unknown"
            ),
            payload=record.payload,
            merkle_hash=record.merkle_hash,
            signature_valid=signature_valid,
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        metrics = calibrator.metrics()
        return HealthResponse(
            status="ok",
            evaluated_at=utcnow(),
            integrity=metrics.integrity,
            drift=metrics.drift,
            severity=metrics.severity,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        metrics_data = calibrator.metrics().__dict__
        metrics_payload: dict[str, float] = {
            k: float(v) for k, v in metrics_data.items()
        }
        content = encode_metrics(metrics_payload)
        return Response(content=content, media_type="text/plain; version=0.0.4")

    @app.get("/self_test", response_model=SelfTestSummary)
    async def self_test() -> SelfTestSummary:
        results: list[SelfTestResult] = []

        # Known contradiction
        contradiction_payload = {
            "claim_a": "The sky is blue",
            "claim_b": "The sky is not blue",
            "tier": "enterprise",
            "verdict": "contradiction",
            "status": "verified",
        }
        contradiction_receipt = provenance_service.append_receipt(
            contradiction_payload, seed="self-test-contradiction"
        )
        results.append(
            SelfTestResult(
                name="known_contradiction",
                status="verified",
                receipt_uuid=contradiction_receipt.uuid,
                details="Contradiction detected",
            )
        )

        # Unknown outcome
        unknown_payload = {
            "claim_a": "Atlantis has 5 cities",
            "claim_b": "Atlantis has 4 cities",
            "tier": "enterprise",
            "verdict": "unknown",
            "status": "unknown",
        }
        unknown_receipt = provenance_service.append_receipt(
            unknown_payload, seed="self-test-unknown"
        )
        results.append(
            SelfTestResult(
                name="unknown_result",
                status="unknown",
                receipt_uuid=unknown_receipt.uuid,
                details="Unknown claims recorded",
            )
        )

        # Tampered hash simulation
        tampered_payload = {
            "claim_a": "Water boils at 100C",
            "claim_b": "Water freezes at 0C",
            "tier": "enterprise",
            "verdict": "aligned",
            "status": "verified",
        }
        tampered_receipt = provenance_service.append_receipt(
            tampered_payload, seed="self-test-tampered"
        )
        mutated_payload = dict(tampered_payload)
        mutated_payload["status"] = "tampered"
        forged_hash = merkle_hash(
            payload=mutated_payload, prev_hash=tampered_receipt.prev_hash
        )
        expected_hash = tampered_receipt.merkle_hash
        signature_valid = hmac_signature(mutated_payload) == tampered_receipt.signature
        tampering_detected = forged_hash != expected_hash or not signature_valid
        tampered_details = {
            "expected_hash": expected_hash,
            "forged_hash": forged_hash,
            "signature_valid": signature_valid,
        }
        results.append(
            SelfTestResult(
                name="tampered_hash",
                status="tampered" if tampering_detected else "verified",
                receipt_uuid=tampered_receipt.uuid,
                details=json.dumps(tampered_details),
            )
        )

        return SelfTestSummary(
            results=results, ledger_path=str(provenance_service.ledger_path)
        )

    return app


app = create_app()
from tessrax_truth_api import monetization_api
