"""Async FastAPI application exposing the Tessrax Truth API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from config_loader import load_config as load_core_config

from tessrax.api.human_feedback import router as human_feedback_router
from tessrax.ledger import Ledger
from tessrax.logging import LedgerWriter, S3LedgerWriter
from tessrax.metabolism.async_detector import AsyncContradictionDetector
from tessrax.tessrax_engine import calculate_stability, route_to_governance_lane
from tessrax.types import Claim

from tessrax_truth_api import monetization_api
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
from tessrax_truth_api.services.cache_service import CachedEntry, CacheService
from tessrax_truth_api.services.entitlement_service import EntitlementService
from tessrax_truth_api.services.provenance_service import ProvenanceService
from tessrax_truth_api.services.stripe_gateway import StripeGateway
from tessrax_truth_api.services.subscription_service import SubscriptionService
from tessrax_truth_api.services.validation_service import ValidationService
from tessrax_truth_api.services.webhook_service import WebhookService
from tessrax_truth_api.utils import (
    hmac_signature,
    issue_jwt,
    load_config,
    merkle_hash,
    utcnow,
    verify_signature,
)


class AgentClaim(BaseModel):
    """Payload representing a free-form claim submitted by an agent."""

    agent: str = Field(..., description="Agent identifier")
    claim: str = Field(..., description="Agent claim text")
    context: dict[str, Any] | None = Field(
        default=None, description="Optional claim metadata"
    )


class ClaimsSubmission(BaseModel):
    """Container for a batch of agent claims."""

    claims: list[AgentClaim] = Field(..., description="Claims to analyse")


class AnalysisResult(BaseModel):
    """Response structure for stability analysis requests."""

    stability_score: float = Field(..., ge=0.0, le=1.0)
    governance_lane: str


class StructuredClaim(BaseModel):
    """Structured payload used for asynchronous contradiction detection."""

    claim_id: str
    subject: str
    metric: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    context: dict[str, str] = Field(default_factory=dict)

    def to_claim(self) -> Claim:
        return Claim(
            claim_id=self.claim_id,
            subject=self.subject,
            metric=self.metric,
            value=self.value,
            unit=self.unit,
            timestamp=self.timestamp,
            source=self.source,
            context=dict(self.context),
        )


def create_app() -> FastAPI:
    config = load_config()
    core_config = load_core_config()
    app = FastAPI(title="Tessrax Truth API", version="2.0.0")

    metrics_registry = CollectorRegistry()
    requests_total = Counter(
        "tessrax_requests_total",
        "Total Tessrax Truth API requests",
        registry=metrics_registry,
    )
    errors_total = Counter(
        "tessrax_errors_total",
        "Total Tessrax Truth API error responses",
        registry=metrics_registry,
    )
    latency_histogram = Histogram(
        "tessrax_latency_seconds",
        "Tessrax Truth API latency in seconds",
        registry=metrics_registry,
    )
    integrity_gauge = Gauge(
        "truth_api_integrity",
        "Current integrity calibration value",
        registry=metrics_registry,
    )
    drift_gauge = Gauge(
        "truth_api_drift",
        "Current drift calibration value",
        registry=metrics_registry,
    )
    severity_gauge = Gauge(
        "truth_api_severity",
        "Current severity calibration value",
        registry=metrics_registry,
    )

    ledger_path = Path(core_config.logging.ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    cloud_writer = None
    if core_config.logging.cloud and core_config.logging.cloud.enabled:
        cloud_writer = S3LedgerWriter(core_config.logging.cloud)
    ledger_writer = LedgerWriter(ledger_path, cloud_writer)

    governance_ledger = Ledger()
    detector = AsyncContradictionDetector(governance_ledger)

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
    app.include_router(human_feedback_router)

    app.add_middleware(TruthLockMiddleware)

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        requests_total.inc()
        with latency_histogram.time():
            response = await call_next(request)
        if response.status_code >= 400:
            errors_total.inc()
        return response

    @app.on_event("startup")
    async def startup_event() -> None:
        await detector.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        await detector.stop()
        ledger_writer.close()

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

    @app.post("/submit_claims", response_model=AnalysisResult)
    async def submit_claims(submission: ClaimsSubmission) -> AnalysisResult:
        if not submission.claims:
            raise HTTPException(status_code=400, detail="At least one claim is required")

        raw_claims = [claim.dict() for claim in submission.claims]
        stability = calculate_stability(raw_claims)
        lane = route_to_governance_lane(stability, core_config.thresholds)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stability_score": stability,
            "governance_lane": lane,
            "claims": raw_claims,
        }

        ledger_writer.append(record)

        return AnalysisResult(stability_score=stability, governance_lane=lane)

    @app.get("/ledger")
    async def get_ledger() -> JSONResponse:
        if not ledger_path.exists():
            return JSONResponse(content=[], status_code=200)

        records: list[dict[str, Any]] = []
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise HTTPException(
                        status_code=500, detail=f"Invalid ledger entry: {exc}"
                    )
        return JSONResponse(content=records)

    @app.post("/claims")
    async def submit_structured_claims(claims: list[StructuredClaim]) -> dict[str, object]:
        if not claims:
            raise HTTPException(status_code=400, detail="At least one claim is required")

        for payload in claims:
            await detector.publish(payload.to_claim())
        return {"status": "queued", "count": len(claims)}

    @app.get("/contradictions/live")
    async def get_live_contradictions() -> dict[str, int]:
        return {"active": len(detector.seen)}

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
        calibration_metrics = calibrator.metrics()
        integrity_gauge.set(float(calibration_metrics.integrity))
        drift_gauge.set(float(calibration_metrics.drift))
        severity_gauge.set(float(calibration_metrics.severity))

        content = generate_latest(metrics_registry)
        return Response(content=content, media_type=CONTENT_TYPE_LATEST)

    @app.get("/self_test", response_model=SelfTestSummary)
    async def self_test() -> SelfTestSummary:
        results: list[SelfTestResult] = []

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
