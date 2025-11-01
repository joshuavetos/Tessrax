"""
Tessrax Monetization Layer Tests
Module: tests.test_monetization
Governance Clauses: AEP-001, RVC-001, POST-AUDIT-001

Deterministic tests for subscription lifecycle, entitlements, and billing.
All tests emit governance receipts with integrity ≥ 0.94.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from tessrax_truth_api.services.subscription_service import (
    SubscriptionService,
    SubscriptionTier,
    SubscriptionStatus
)
from tessrax_truth_api.services.entitlement_service import EntitlementService
from tessrax_truth_api.services.webhook_service import WebhookService, WebhookEvent


# Fixtures

@pytest.fixture
def temp_ledger():
    """Create temporary ledger file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        ledger_path = Path(f.name)
    yield ledger_path
    ledger_path.unlink(missing_ok=True)


@pytest.fixture
def temp_events():
    """Create temporary webhook events file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        events_path = Path(f.name)
    yield events_path
    events_path.unlink(missing_ok=True)


@pytest.fixture
def subscription_service(temp_ledger):
    """Initialize SubscriptionService with temporary ledger."""
    return SubscriptionService(ledger_path=temp_ledger)


@pytest.fixture
def entitlement_service(subscription_service):
    """Initialize EntitlementService."""
    return EntitlementService(subscription_service)


@pytest.fixture
def webhook_service(subscription_service, temp_events):
    """Initialize WebhookService."""
    return WebhookService(
        subscription_service=subscription_service,
        processed_events_path=temp_events
    )


# Subscription Lifecycle Tests

class TestSubscriptionLifecycle:
    """Test all subscription lifecycle state transitions."""

    def test_create_subscription_starter(self, subscription_service, temp_ledger):
        """Test creating starter tier subscription."""
        sub, receipt = subscription_service.create(
            customer_id="test_customer_001",
            tier=SubscriptionTier.STARTER,
            trial_days=0
        )

        # Validate subscription
        assert sub.tier == SubscriptionTier.STARTER
        assert sub.status == SubscriptionStatus.ACTIVE
        assert sub.quota_limit == 10_000
        assert sub.usage_count == 0
        assert sub.customer_id == "test_customer_001"

        # Validate receipt
        assert receipt.integrity_score >= 0.94
        assert receipt.event_type == "subscription.created"
        assert receipt.governance_lane == "general_lane"
        assert "AEP-001" in receipt.clauses
        assert receipt.amount == 49.00

        # Validate ledger persistence
        assert temp_ledger.exists()
        with open(temp_ledger, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["subscription_id"] == sub.subscription_id
            assert entry["integrity_score"] >= 0.94

    def test_create_subscription_with_trial(self, subscription_service):
        """Test creating subscription with trial period."""
        sub, receipt = subscription_service.create(
            customer_id="test_customer_002",
            tier=SubscriptionTier.PROFESSIONAL,
            trial_days=14
        )

        assert sub.status == SubscriptionStatus.TRIALING
        assert sub.trial_end is not None
        assert receipt.compliance_metadata["trial_days"] == 14

    def test_upgrade_subscription(self, subscription_service):
        """Test upgrading from starter to professional."""
        # Create starter subscription
        sub, _ = subscription_service.create(
            customer_id="test_customer_003",
            tier=SubscriptionTier.STARTER
        )

        # Upgrade to professional
        sub_upgraded, receipt = subscription_service.upgrade(
            subscription_id=sub.subscription_id,
            new_tier=SubscriptionTier.PROFESSIONAL
        )

        assert sub_upgraded.tier == SubscriptionTier.PROFESSIONAL
        assert sub_upgraded.quota_limit == 100_000
        assert sub_upgraded.usage_count == 0  # Reset on upgrade
        assert receipt.event_type == "subscription.upgraded"
        assert receipt.compliance_metadata["old_tier"] == "starter"
        assert receipt.compliance_metadata["new_tier"] == "professional"
        assert receipt.integrity_score >= 0.94

    def test_downgrade_subscription(self, subscription_service):
        """Test downgrading from enterprise to professional."""
        # Create enterprise subscription
        sub, _ = subscription_service.create(
            customer_id="test_customer_004",
            tier=SubscriptionTier.ENTERPRISE
        )

        # Downgrade to professional
        sub_downgraded, receipt = subscription_service.downgrade(
            subscription_id=sub.subscription_id,
            new_tier=SubscriptionTier.PROFESSIONAL
        )

        assert sub_downgraded.tier == SubscriptionTier.PROFESSIONAL
        assert sub_downgraded.quota_limit == 100_000
        assert receipt.event_type == "subscription.downgraded"
        assert receipt.integrity_score >= 0.94

    def test_cancel_subscription(self, subscription_service):
        """Test subscription cancellation."""
        # Create subscription
        sub, _ = subscription_service.create(
            customer_id="test_customer_005",
            tier=SubscriptionTier.PROFESSIONAL
        )

        # Cancel
        sub_canceled, receipt = subscription_service.cancel(
            subscription_id=sub.subscription_id,
            reason="test_cancellation"
        )

        assert sub_canceled.status == SubscriptionStatus.CANCELED
        assert sub_canceled.canceled_at is not None
        assert receipt.event_type == "subscription.canceled"
        assert receipt.compliance_metadata["cancellation_reason"] == "test_cancellation"
        assert receipt.governance_lane == "general_lane"

    def test_renew_subscription_success(self, subscription_service):
        """Test successful subscription renewal."""
        # Create subscription
        sub, _ = subscription_service.create(
            customer_id="test_customer_006",
            tier=SubscriptionTier.PROFESSIONAL
        )

        # Record some usage
        subscription_service.record_usage(sub.subscription_id, count=5000)

        # Renew
        sub_renewed, receipt = subscription_service.renew(
            subscription_id=sub.subscription_id,
            payment_succeeded=True
        )

        assert sub_renewed.status == SubscriptionStatus.ACTIVE
        assert sub_renewed.usage_count == 0  # Reset on renewal
        assert receipt.event_type == "subscription.renewed"
        assert receipt.status == "success"

    def test_renew_subscription_failed(self, subscription_service):
        """Test failed subscription renewal (payment failed)."""
        # Create subscription
        sub, _ = subscription_service.create(
            customer_id="test_customer_007",
            tier=SubscriptionTier.STARTER
        )

        # Renew with payment failure
        sub_past_due, receipt = subscription_service.renew(
            subscription_id=sub.subscription_id,
            payment_succeeded=False
        )

        assert sub_past_due.status == SubscriptionStatus.PAST_DUE
        assert receipt.event_type == "payment.failed"
        assert receipt.status == "failed"
        assert receipt.governance_lane == "review_lane"  # Payment failures go to review


# Usage Tracking Tests

class TestUsageTracking:
    """Test quota enforcement and usage metering."""

    def test_record_usage_within_quota(self, subscription_service):
        """Test recording usage within quota limits."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_008",
            tier=SubscriptionTier.STARTER  # 10,000 limit
        )

        # Record usage
        sub_used = subscription_service.record_usage(sub.subscription_id, count=5000)
        assert sub_used.usage_count == 5000

        # Record more
        sub_used = subscription_service.record_usage(sub.subscription_id, count=3000)
        assert sub_used.usage_count == 8000

    def test_record_usage_exceed_quota(self, subscription_service):
        """Test quota exceeded error."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_009",
            tier=SubscriptionTier.STARTER
        )

        # Use up quota
        subscription_service.record_usage(sub.subscription_id, count=10_000)

        # Attempt to exceed
        with pytest.raises(ValueError, match="Quota exceeded"):
            subscription_service.record_usage(sub.subscription_id, count=1)

    def test_record_usage_unlimited_quota(self, subscription_service):
        """Test enterprise tier with unlimited quota."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_010",
            tier=SubscriptionTier.ENTERPRISE
        )

        # Record massive usage (no limit)
        sub_used = subscription_service.record_usage(sub.subscription_id, count=1_000_000)
        assert sub_used.usage_count == 1_000_000
        assert sub_used.quota_limit is None


# Entitlement Tests

class TestEntitlementService:
    """Test subscription-based access control."""

    def test_check_entitlement_allowed(self, subscription_service, entitlement_service):
        """Test allowed entitlement check."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_011",
            tier=SubscriptionTier.PROFESSIONAL
        )
        sub.status = SubscriptionStatus.ACTIVE

        check = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )

        assert check.allowed is True
        assert check.tier == "professional"
        assert check.quota_remaining == 100_000
        assert "contradiction_detection" in check.features

    def test_check_entitlement_quota_exceeded(self, subscription_service, entitlement_service):
        """Test entitlement denied due to quota."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_012",
            tier=SubscriptionTier.STARTER
        )
        sub.status = SubscriptionStatus.ACTIVE

        # Exhaust quota
        subscription_service.record_usage(sub.subscription_id, count=10_000)

        check = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )

        assert check.allowed is False
        assert check.reason == "quota_exceeded"
        assert check.quota_remaining == 0

    def test_check_entitlement_feature_not_in_tier(self, subscription_service, entitlement_service):
        """Test entitlement denied due to missing feature."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_013",
            tier=SubscriptionTier.STARTER
        )
        sub.status = SubscriptionStatus.ACTIVE

        check = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="custom_policy_compiler",  # Enterprise only
            usage_count=1
        )

        assert check.allowed is False
        assert "feature_not_in_tier" in check.reason

    def test_check_entitlement_canceled_subscription(self, subscription_service, entitlement_service):
        """Test entitlement denied for canceled subscription."""
        sub, _ = subscription_service.create(
            customer_id="test_customer_014",
            tier=SubscriptionTier.PROFESSIONAL
        )

        # Cancel subscription
        subscription_service.cancel(sub.subscription_id)

        check = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection",
            usage_count=1
        )

        assert check.allowed is False
        assert check.reason == "subscription_canceled"


# Webhook Tests

class TestWebhookService:
    """Test Stripe webhook event processing."""

    def test_checkout_completed(self, webhook_service, subscription_service):
        """Test checkout.session.completed event."""
        event = WebhookEvent(
            event_id="evt_test_001",
            event_type="checkout.session.completed",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "customer": "cus_test_001",
                    "subscription": "sub_stripe_001",
                    "metadata": {"tier": "professional"}
                }
            }
        )

        result = webhook_service.process_event(event)

        assert result["status"] == "processed"
        assert "subscription_id" in result["result"]
        assert result["result"]["tier"] == "professional"

        # Verify subscription created
        subscriptions = subscription_service.list_subscriptions()
        assert len(subscriptions) == 1
        assert subscriptions[0].tier == SubscriptionTier.PROFESSIONAL

    def test_payment_succeeded(self, webhook_service, subscription_service):
        """Test invoice.payment_succeeded event."""
        # First create subscription via checkout
        checkout_event = WebhookEvent(
            event_id="evt_test_002",
            event_type="checkout.session.completed",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "customer": "cus_test_002",
                    "subscription": "sub_stripe_002",
                    "metadata": {"tier": "starter"}
                }
            }
        )
        webhook_service.process_event(checkout_event)

        # Then process payment succeeded
        payment_event = WebhookEvent(
            event_id="evt_test_003",
            event_type="invoice.payment_succeeded",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "subscription": "sub_stripe_002"
                }
            }
        )

        result = webhook_service.process_event(payment_event)

        assert result["status"] == "processed"
        assert result["result"]["status"] == "renewed"

    def test_webhook_idempotency(self, webhook_service):
        """Test webhook idempotency (duplicate event handling)."""
        event = WebhookEvent(
            event_id="evt_test_idempotent",
            event_type="checkout.session.completed",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "customer": "cus_test_003",
                    "subscription": "sub_stripe_003",
                    "metadata": {"tier": "professional"}
                }
            }
        )

        # Process first time
        result1 = webhook_service.process_event(event)
        assert result1["status"] == "processed"

        # Process again (duplicate)
        result2 = webhook_service.process_event(event)
        assert result2["status"] == "already_processed"

    def test_webhook_signature_verification(self, webhook_service):
        """Test Stripe webhook signature verification."""
        import hmac
        import hashlib

        payload = b'{"event": "test"}'
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        signed_payload = f"{timestamp}.{payload.decode()}"
        signature = hmac.new(
            webhook_service.stripe_webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        signature_header = f"t={timestamp},v1={signature}"

        is_valid = webhook_service.verify_signature(payload, signature_header)
        assert is_valid is True

        # Test invalid signature
        invalid_sig = webhook_service.verify_signature(payload, "t=123,v1=invalid")
        assert invalid_sig is False


# Governance Receipt Tests

class TestGovernanceReceipts:
    """Test governance receipt generation and integrity."""

    def test_receipt_integrity_scores(self, subscription_service):
        """Test all receipts meet minimum integrity requirement."""
        # Create subscription
        _, receipt1 = subscription_service.create(
            customer_id="test_customer_015",
            tier=SubscriptionTier.PROFESSIONAL
        )
        assert receipt1.integrity_score >= 0.94

        # Upgrade
        sub = list(subscription_service.list_subscriptions())[0]
        _, receipt2 = subscription_service.upgrade(
            subscription_id=sub.subscription_id,
            new_tier=SubscriptionTier.ENTERPRISE
        )
        assert receipt2.integrity_score >= 0.94

        # Renew
        _, receipt3 = subscription_service.renew(
            subscription_id=sub.subscription_id,
            payment_succeeded=True
        )
        assert receipt3.integrity_score >= 0.94

        # Cancel
        _, receipt4 = subscription_service.cancel(sub.subscription_id)
        assert receipt4.integrity_score >= 0.94

    def test_receipt_compliance_metadata(self, subscription_service):
        """Test all receipts include required compliance metadata."""
        _, receipt = subscription_service.create(
            customer_id="test_customer_016",
            tier=SubscriptionTier.PROFESSIONAL
        )

        compliance = receipt.compliance_metadata

        # Required fields
        assert "pci_dss_exemption" in compliance
        assert "gdpr_processing_basis" in compliance
        assert "data_retention_days" in compliance
        assert "privacy_mode" in compliance

        # Values
        assert compliance["pci_dss_exemption"] == "no_card_data_stored"
        assert compliance["gdpr_processing_basis"] == "contract_performance"
        assert compliance["data_retention_days"] == 2555  # 7 years
        assert compliance["privacy_mode"] == "pseudonymized"

    def test_receipt_governance_lanes(self, subscription_service):
        """Test receipts routed to correct governance lanes."""
        # Normal operations → general_lane
        _, receipt_create = subscription_service.create(
            customer_id="test_customer_017",
            tier=SubscriptionTier.STARTER
        )
        assert receipt_create.governance_lane == "general_lane"

        # Payment failure → review_lane
        sub = list(subscription_service.list_subscriptions())[0]
        _, receipt_failed = subscription_service.renew(
            subscription_id=sub.subscription_id,
            payment_succeeded=False
        )
        assert receipt_failed.governance_lane == "review_lane"

    def test_receipt_signature_validity(self, subscription_service):
        """Test receipt HMAC signatures."""
        _, receipt = subscription_service.create(
            customer_id="test_customer_018",
            tier=SubscriptionTier.PROFESSIONAL
        )

        # Validate signature present
        assert receipt.signature is not None
        assert len(receipt.signature) == 64  # SHA256 hex

        # Validate signature is deterministic
        import hashlib
        import hmac

        receipt_data = {
            "receipt_id": receipt.receipt_id,
            "module": receipt.module,
            "event_type": receipt.event_type,
            "subscription_id": receipt.subscription_id,
            "customer_id": receipt.customer_id,
            "tier": receipt.tier,
            "amount": receipt.amount,
            "status": receipt.status,
            "metrics": receipt.metrics,
            "integrity_score": receipt.integrity_score,
            "merkle_root": receipt.merkle_root,
            "merkle_proof": receipt.merkle_proof,
            "governance_lane": receipt.governance_lane,
            "compliance_metadata": receipt.compliance_metadata,
            "timestamp": receipt.timestamp
        }

        canonical = json.dumps(receipt_data, sort_keys=True, separators=(",", ":"))
        expected_signature = hmac.new(
            subscription_service.hmac_secret.encode(),
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        assert receipt.signature == expected_signature


# Integration Tests

class TestEndToEndIntegration:
    """Test complete subscription lifecycle end-to-end."""

    def test_full_subscription_lifecycle(self, subscription_service, entitlement_service):
        """Test complete lifecycle: create → use → upgrade → renew → cancel."""
        # 1. Create subscription
        sub, receipt1 = subscription_service.create(
            customer_id="test_customer_e2e",
            tier=SubscriptionTier.STARTER,
            trial_days=14
        )
        assert sub.status == SubscriptionStatus.TRIALING
        assert receipt1.integrity_score >= 0.94

        # 2. Check entitlement during trial
        sub.status = SubscriptionStatus.ACTIVE  # Activate for testing
        check1 = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection"
        )
        assert check1.allowed is True

        # 3. Record usage
        subscription_service.record_usage(sub.subscription_id, count=5000)
        check2 = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection"
        )
        assert check2.quota_remaining == 5000

        # 4. Upgrade to professional
        sub, receipt2 = subscription_service.upgrade(
            subscription_id=sub.subscription_id,
            new_tier=SubscriptionTier.PROFESSIONAL
        )
        assert sub.tier == SubscriptionTier.PROFESSIONAL
        assert sub.usage_count == 0  # Reset
        assert receipt2.integrity_score >= 0.94

        # 5. Renew subscription
        sub, receipt3 = subscription_service.renew(
            subscription_id=sub.subscription_id,
            payment_succeeded=True
        )
        assert sub.status == SubscriptionStatus.ACTIVE
        assert receipt3.integrity_score >= 0.94

        # 6. Cancel subscription
        sub, receipt4 = subscription_service.cancel(sub.subscription_id)
        assert sub.status == SubscriptionStatus.CANCELED
        assert receipt4.integrity_score >= 0.94

        # 7. Check entitlement after cancellation
        check3 = entitlement_service.check_entitlement(
            subscription_id=sub.subscription_id,
            feature="contradiction_detection"
        )
        assert check3.allowed is False
        assert check3.reason == "subscription_canceled"


# Run self-tests from services
def test_subscription_service_self_test():
    """Run SubscriptionService internal self-test."""
    from tessrax_truth_api.services.subscription_service import _self_test
    assert _self_test() is True


def test_entitlement_service_self_test():
    """Run EntitlementService internal self-test."""
    from tessrax_truth_api.services.entitlement_service import _self_test
    assert _self_test() is True


def test_webhook_service_self_test():
    """Run WebhookService internal self-test."""
    from tessrax_truth_api.services.webhook_service import _self_test
    assert _self_test() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
