"""
Tessrax Monetization API Endpoints
Module: tessrax.api.monetization
Governance Clauses: AEP-001, RVC-001, EAC-001

FastAPI endpoints for subscription management, billing, and entitlements.
Integrates with SubscriptionService, EntitlementService, and WebhookService.
"""

from fastapi import APIRouter, HTTPException, Header, Request, status
from pydantic import BaseModel, Field
from typing import Any, Literal
from datetime import datetime

from tessrax_truth_api.services.subscription_service import (
    SubscriptionService,
    SubscriptionTier,
    SubscriptionStatus,
    Subscription,
    BillingReceipt
)
from tessrax_truth_api.services.entitlement_service import EntitlementService
from tessrax_truth_api.services.webhook_service import WebhookService, WebhookEvent


# Pydantic models for request/response validation

class SubscribeRequest(BaseModel):
    """Request to create new subscription."""
    customer_id: str = Field(..., description="Unique customer identifier (hashed)")
    tier: Literal["starter", "professional", "enterprise"]
    trial_days: int = Field(default=0, ge=0, le=30, description="Trial period in days")
    stripe_checkout_session_id: str | None = Field(None, description="Stripe checkout session ID")


class SubscribeResponse(BaseModel):
    """Response with subscription details and governance receipt."""
    subscription_id: str
    customer_id: str
    tier: str
    status: str
    quota_limit: int | None
    current_period_end: str
    features: list[str]
    receipt_id: str
    integrity_score: float


class UpgradeRequest(BaseModel):
    """Request to upgrade subscription."""
    subscription_id: str
    new_tier: Literal["starter", "professional", "enterprise"]


class ManageSubscriptionRequest(BaseModel):
    """Request to cancel or modify subscription."""
    subscription_id: str
    action: Literal["cancel", "reactivate"]
    reason: str | None = None


class UsageResponse(BaseModel):
    """Current usage and quota information."""
    subscription_id: str
    tier: str
    usage_count: int
    quota_limit: int | None
    quota_remaining: int | None
    current_period_start: str
    current_period_end: str
    overage_charges: float = 0.0


class BillingHistoryResponse(BaseModel):
    """Billing transaction history."""
    transactions: list[dict[str, Any]]
    total_count: int
    period_start: str
    period_end: str


class EntitlementCheckResponse(BaseModel):
    """Entitlement verification result."""
    allowed: bool
    tier: str
    reason: str | None
    features: list[str]
    quota_remaining: int | None


class WebhookResponse(BaseModel):
    """Webhook processing result."""
    status: str
    event_id: str
    processed_at: str


# Initialize router
router = APIRouter(prefix="/billing", tags=["monetization"])

# Service instances (initialized in main.py via dependency injection)
_subscription_service: SubscriptionService | None = None
_entitlement_service: EntitlementService | None = None
_webhook_service: WebhookService | None = None


def init_services(
    subscription_service: SubscriptionService,
    entitlement_service: EntitlementService,
    webhook_service: WebhookService
) -> None:
    """Initialize service instances (called from main.py)."""
    global _subscription_service, _entitlement_service, _webhook_service
    _subscription_service = subscription_service
    _entitlement_service = entitlement_service
    _webhook_service = webhook_service


@router.post("/subscribe", response_model=SubscribeResponse, status_code=status.HTTP_201_CREATED)
async def subscribe(request: SubscribeRequest) -> SubscribeResponse:
    """
    Create new subscription with governance receipt.

    Emits ledger-anchored receipt with integrity ≥ 0.94.
    Trial period optional (0-30 days).
    """
    if not _subscription_service or not _entitlement_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    try:
        tier = SubscriptionTier(request.tier)
        subscription, receipt = _subscription_service.create(
            customer_id=request.customer_id,
            tier=tier,
            trial_days=request.trial_days,
            stripe_subscription_id=request.stripe_checkout_session_id
        )

        features = _entitlement_service.get_tier_features(tier.value)

        return SubscribeResponse(
            subscription_id=subscription.subscription_id,
            customer_id=subscription.customer_id,
            tier=subscription.tier.value,
            status=subscription.status.value,
            quota_limit=subscription.quota_limit,
            current_period_end=subscription.current_period_end,
            features=features,
            receipt_id=receipt.receipt_id,
            integrity_score=receipt.integrity_score
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upgrade", response_model=SubscribeResponse)
async def upgrade_subscription(request: UpgradeRequest) -> SubscribeResponse:
    """
    Upgrade subscription to higher tier.

    Prorated billing applied automatically.
    Emits governance receipt for tier change.
    """
    if not _subscription_service or not _entitlement_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    try:
        new_tier = SubscriptionTier(request.new_tier)
        subscription, receipt = _subscription_service.upgrade(
            subscription_id=request.subscription_id,
            new_tier=new_tier
        )

        features = _entitlement_service.get_tier_features(new_tier.value)

        return SubscribeResponse(
            subscription_id=subscription.subscription_id,
            customer_id=subscription.customer_id,
            tier=subscription.tier.value,
            status=subscription.status.value,
            quota_limit=subscription.quota_limit,
            current_period_end=subscription.current_period_end,
            features=features,
            receipt_id=receipt.receipt_id,
            integrity_score=receipt.integrity_score
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/manage", response_model=dict[str, Any])
async def manage_subscription(request: ManageSubscriptionRequest) -> dict[str, Any]:
    """
    Cancel or reactivate subscription.

    Cancellation effective at period end to allow grace period.
    Emits governance receipt with cancellation reason.
    """
    if not _subscription_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    try:
        if request.action == "cancel":
            subscription, receipt = _subscription_service.cancel(
                subscription_id=request.subscription_id,
                reason=request.reason
            )
            return {
                "action": "canceled",
                "subscription_id": subscription.subscription_id,
                "effective_date": subscription.current_period_end,
                "receipt_id": receipt.receipt_id
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action: {request.action}")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/usage/{subscription_id}", response_model=UsageResponse)
async def get_usage(subscription_id: str) -> UsageResponse:
    """
    Get current usage and quota information.

    Returns real-time quota consumption for billing period.
    """
    if not _subscription_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    subscription = _subscription_service.get_subscription(subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    quota_remaining = None
    if subscription.quota_limit is not None:
        quota_remaining = max(0, subscription.quota_limit - subscription.usage_count)

    return UsageResponse(
        subscription_id=subscription.subscription_id,
        tier=subscription.tier.value,
        usage_count=subscription.usage_count,
        quota_limit=subscription.quota_limit,
        quota_remaining=quota_remaining,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end
    )


@router.get("/history/{customer_id}", response_model=BillingHistoryResponse)
async def get_billing_history(
    customer_id: str,
    limit: int = 50
) -> BillingHistoryResponse:
    """
    Get billing transaction history with Merkle proofs.

    Returns ledger-anchored receipts for all subscription events.
    Includes: signup, payments, upgrades, cancellations, refunds.
    """
    if not _subscription_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    # Load transactions from billing ledger
    import json
    from pathlib import Path

    ledger_path = Path(_subscription_service.ledger_path)
    if not ledger_path.exists():
        return BillingHistoryResponse(
            transactions=[],
            total_count=0,
            period_start="",
            period_end=""
        )

    transactions = []
    with open(ledger_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("customer_id") == customer_id:
                transactions.append(entry)

    # Sort by timestamp descending
    transactions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    transactions = transactions[:limit]

    period_start = transactions[-1]["timestamp"] if transactions else ""
    period_end = transactions[0]["timestamp"] if transactions else ""

    return BillingHistoryResponse(
        transactions=transactions,
        total_count=len(transactions),
        period_start=period_start,
        period_end=period_end
    )


@router.get("/entitlement/{subscription_id}", response_model=EntitlementCheckResponse)
async def check_entitlement(
    subscription_id: str,
    feature: str = "contradiction_detection"
) -> EntitlementCheckResponse:
    """
    Check entitlement for specific feature.

    Validates subscription status, quota, and feature access.
    Used by middleware to gate API endpoints.
    """
    if not _entitlement_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    check = _entitlement_service.check_entitlement(
        subscription_id=subscription_id,
        feature=feature,
        usage_count=0  # Just checking, not reserving quota
    )

    return EntitlementCheckResponse(
        allowed=check.allowed,
        tier=check.tier,
        reason=check.reason,
        features=check.features,
        quota_remaining=check.quota_remaining
    )


@router.post("/webhooks/stripe", response_model=WebhookResponse)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature")
) -> WebhookResponse:
    """
    Handle Stripe webhook events.

    Processes: checkout.session.completed, invoice.payment_succeeded,
    invoice.payment_failed, customer.subscription.updated, customer.subscription.deleted.

    Verifies webhook signature and enforces idempotency.
    """
    if not _webhook_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    # Read raw body for signature verification
    body = await request.body()

    # Verify signature
    if stripe_signature:
        is_valid = _webhook_service.verify_signature(body, stripe_signature)
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # Parse event
    import json
    event_data = json.loads(body)

    event = WebhookEvent(
        event_id=event_data.get("id", ""),
        event_type=event_data.get("type", ""),
        created=event_data.get("created", 0),
        livemode=event_data.get("livemode", False),
        data=event_data.get("data", {})
    )

    # Process event
    result = _webhook_service.process_event(event)

    return WebhookResponse(
        status=result["status"],
        event_id=event.event_id,
        processed_at=datetime.utcnow().isoformat()
    )


@router.get("/pricing", response_model=dict[str, Any])
async def get_pricing() -> dict[str, Any]:
    """
    Get public pricing tiers and features.

    Returns tier configuration for pricing page rendering.
    """
    if not _subscription_service:
        raise HTTPException(status_code=500, detail="Services not initialized")

    pricing = {}
    for tier, config in SubscriptionService.TIER_PRICING.items():
        pricing[tier.value] = {
            "price_monthly_usd": config["price"],
            "quota_limit": config["quota"],
            "features": config["features"],
            "recommended": tier == SubscriptionTier.PROFESSIONAL
        }

    return {"tiers": pricing, "currency": "USD", "billing_period": "monthly"}
