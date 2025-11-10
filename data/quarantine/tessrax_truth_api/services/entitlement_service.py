"""
Tessrax Entitlement Service
Module: tessrax.billing.entitlement_service
Governance Clauses: AEP-001, RVC-001, EAC-001

Enforces subscription-based access control and quota limits.
Integrates with SubscriptionService and JWT middleware.
"""

from typing import Literal
from dataclasses import dataclass
from datetime import datetime, timezone
from tessrax_truth_api.services.subscription_service import (
    SubscriptionService,
    SubscriptionStatus,
    SubscriptionTier
)


@dataclass
class EntitlementCheck:
    """Result of entitlement verification."""
    allowed: bool
    subscription_id: str | None
    tier: str
    reason: str | None
    quota_remaining: int | None
    features: list[str]


class EntitlementService:
    """
    Enforces subscription-based entitlements and feature gates.

    Checks subscription status, quota limits, and feature access
    before allowing API operations.
    """

    # Feature gates per tier
    TIER_FEATURES = {
        "starter": [
            "contradiction_detection",
            "basic_dashboard",
            "email_support",
            "7_day_retention"
        ],
        "professional": [
            "contradiction_detection",
            "basic_dashboard",
            "advanced_analytics",
            "api_access",
            "priority_support",
            "90_day_retention",
            "export_receipts"
        ],
        "enterprise": [
            "contradiction_detection",
            "basic_dashboard",
            "advanced_analytics",
            "api_access",
            "dedicated_kernel",
            "sla_99_9",
            "sso",
            "white_glove_support",
            "1_year_retention",
            "custom_policy_compiler",
            "export_receipts",
            "audit_logs"
        ]
    }

    def __init__(self, subscription_service: SubscriptionService) -> None:
        """
        Initialize entitlement service.

        Args:
            subscription_service: Subscription lifecycle manager
        """
        self.subscription_service = subscription_service

    def check_entitlement(
        self,
        subscription_id: str,
        feature: str = "contradiction_detection",
        usage_count: int = 1
    ) -> EntitlementCheck:
        """
        Check if subscription has access to feature and quota available.

        Args:
            subscription_id: Subscription ID from JWT
            feature: Feature to check access for
            usage_count: Number of quota units to reserve

        Returns:
            EntitlementCheck with access decision and details
        """
        subscription = self.subscription_service.get_subscription(subscription_id)

        if not subscription:
            return EntitlementCheck(
                allowed=False,
                subscription_id=None,
                tier="none",
                reason="subscription_not_found",
                quota_remaining=None,
                features=[]
            )

        tier = subscription.tier.value
        tier_features = self.TIER_FEATURES.get(tier, [])

        # Check subscription status
        if subscription.status == SubscriptionStatus.CANCELED:
            return EntitlementCheck(
                allowed=False,
                subscription_id=subscription_id,
                tier=tier,
                reason="subscription_canceled",
                quota_remaining=0,
                features=tier_features
            )

        if subscription.status == SubscriptionStatus.UNPAID:
            return EntitlementCheck(
                allowed=False,
                subscription_id=subscription_id,
                tier=tier,
                reason="payment_required",
                quota_remaining=0,
                features=tier_features
            )

        # Check feature access
        if feature not in tier_features:
            return EntitlementCheck(
                allowed=False,
                subscription_id=subscription_id,
                tier=tier,
                reason=f"feature_not_in_tier:{feature}",
                quota_remaining=self._calculate_quota_remaining(subscription),
                features=tier_features
            )

        # Check quota (None = unlimited for enterprise)
        if subscription.quota_limit is not None:
            quota_remaining = subscription.quota_limit - subscription.usage_count
            if quota_remaining < usage_count:
                return EntitlementCheck(
                    allowed=False,
                    subscription_id=subscription_id,
                    tier=tier,
                    reason="quota_exceeded",
                    quota_remaining=quota_remaining,
                    features=tier_features
                )
        else:
            quota_remaining = None  # Unlimited

        # Check trial expiration
        if subscription.status == SubscriptionStatus.TRIALING and subscription.trial_end:
            trial_end_dt = datetime.fromisoformat(subscription.trial_end.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > trial_end_dt:
                return EntitlementCheck(
                    allowed=False,
                    subscription_id=subscription_id,
                    tier=tier,
                    reason="trial_expired",
                    quota_remaining=quota_remaining,
                    features=tier_features
                )

        # All checks passed
        return EntitlementCheck(
            allowed=True,
            subscription_id=subscription_id,
            tier=tier,
            reason=None,
            quota_remaining=quota_remaining,
            features=tier_features
        )

    def _calculate_quota_remaining(self, subscription) -> int | None:
        """Calculate remaining quota for subscription."""
        if subscription.quota_limit is None:
            return None  # Unlimited
        return max(0, subscription.quota_limit - subscription.usage_count)

    def has_feature(self, tier: str, feature: str) -> bool:
        """Check if tier includes specific feature."""
        return feature in self.TIER_FEATURES.get(tier, [])

    def get_tier_features(self, tier: str) -> list[str]:
        """Get all features for a tier."""
        return self.TIER_FEATURES.get(tier, [])


def _self_test() -> bool:
    """
    Runtime verification test for EntitlementService (AEP-001/RVC-001).

    Tests quota enforcement, feature gates, and status checks.
    """
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_ledger = Path(f.name)

    try:
        # Setup
        sub_service = SubscriptionService(ledger_path=test_ledger)
        ent_service = EntitlementService(sub_service)

        # Test 1: Create active subscription
        sub, _ = sub_service.create(
            customer_id="test_customer_ent_001",
            tier=SubscriptionTier.PROFESSIONAL
        )
        sub.status = SubscriptionStatus.ACTIVE  # Activate trial

        # Test 2: Check valid entitlement
        check = ent_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )
        assert check.allowed is True
        assert check.tier == "professional"
        assert check.quota_remaining == 100_000

        # Test 3: Record usage and check quota
        sub_service.record_usage(sub.subscription_id, count=99_999)
        check_near_limit = ent_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )
        assert check_near_limit.allowed is True
        assert check_near_limit.quota_remaining == 1

        # Test 4: Exceed quota
        sub_service.record_usage(sub.subscription_id, count=1)
        check_exceeded = ent_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )
        assert check_exceeded.allowed is False
        assert check_exceeded.reason == "quota_exceeded"

        # Test 5: Check feature not in tier
        check_no_feature = ent_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="custom_policy_compiler",  # Enterprise only
            usage_count=1
        )
        assert check_no_feature.allowed is False
        assert "feature_not_in_tier" in check_no_feature.reason

        # Test 6: Canceled subscription
        sub_canceled, _ = sub_service.cancel(sub.subscription_id)
        check_canceled = ent_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )
        assert check_canceled.allowed is False
        assert check_canceled.reason == "subscription_canceled"

        # Test 7: Enterprise unlimited quota
        sub_ent, _ = sub_service.create(
            customer_id="test_customer_ent_002",
            tier=SubscriptionTier.ENTERPRISE
        )
        sub_ent.status = SubscriptionStatus.ACTIVE
        check_unlimited = ent_service.check_entitlement(
            subscription_id=sub_ent.subscription_id,
            feature="contradiction_detection",
            usage_count=1_000_000
        )
        assert check_unlimited.allowed is True
        assert check_unlimited.quota_remaining is None  # Unlimited

        print(f"✓ EntitlementService self-test passed (7 scenarios validated)")
        return True

    finally:
        test_ledger.unlink(missing_ok=True)


if __name__ == "__main__":
    assert _self_test(), "EntitlementService self-test failed"
