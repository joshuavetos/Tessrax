"""
Tessrax Webhook Service
Module: tessrax.billing.webhook_service
Governance Clauses: AEP-001, RVC-001, POST-AUDIT-001, DLK-001

Handles Stripe webhook events for payment processing.
Emits governance receipts for all billing state changes.
"""

import hashlib
import hmac
import json
from typing import Any, Literal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tessrax_truth_api.services.subscription_service import (
    SubscriptionService,
    SubscriptionStatus,
    SubscriptionTier
)


@dataclass
class WebhookEvent:
    """Stripe webhook event structure."""
    event_id: str
    event_type: str
    created: int
    livemode: bool
    data: dict[str, Any]
    signature: str | None = None


class WebhookService:
    """
    Processes Stripe webhook events with idempotency and governance logging.

    Supported events:
    - checkout.session.completed (subscription creation)
    - invoice.payment_succeeded (successful payment)
    - invoice.payment_failed (failed payment)
    - customer.subscription.updated (tier changes)
    - customer.subscription.deleted (cancellations)
    """

    def __init__(
        self,
        subscription_service: SubscriptionService,
        stripe_webhook_secret: str | None = None,
        processed_events_path: Path | str = "ledger/webhook_events.jsonl"
    ) -> None:
        """
        Initialize webhook service.

        Args:
            subscription_service: Subscription lifecycle manager
            stripe_webhook_secret: Stripe webhook signing secret
            processed_events_path: Path to idempotency log
        """
        self.subscription_service = subscription_service
        self.stripe_webhook_secret = stripe_webhook_secret or "whsec_test_secret"
        self.processed_events_path = Path(processed_events_path)
        self.processed_events_path.parent.mkdir(parents=True, exist_ok=True)

        # Load processed event IDs for idempotency
        self._processed_events: set[str] = set()
        self._load_processed_events()

    def _load_processed_events(self) -> None:
        """Load processed event IDs from disk."""
        if not self.processed_events_path.exists():
            return

        with open(self.processed_events_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                event_data = json.loads(line)
                self._processed_events.add(event_data["event_id"])

    def _mark_processed(self, event_id: str, event_type: str, result: str) -> None:
        """Mark event as processed for idempotency."""
        self._processed_events.add(event_id)

        with open(self.processed_events_path, "a") as f:
            f.write(json.dumps({
                "event_id": event_id,
                "event_type": event_type,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "result": result
            }) + "\n")

    def verify_signature(
        self,
        payload: bytes,
        signature_header: str
    ) -> bool:
        """
        Verify Stripe webhook signature.

        Args:
            payload: Raw webhook payload bytes
            signature_header: Stripe-Signature header value

        Returns:
            True if signature valid
        """
        # Parse signature header: t=timestamp,v1=signature
        sig_parts = dict(part.split("=", 1) for part in signature_header.split(","))
        timestamp = sig_parts.get("t")
        signature = sig_parts.get("v1")

        if not timestamp or not signature:
            return False

        # Construct signed payload
        signed_payload = f"{timestamp}.{payload.decode()}"
        expected_signature = hmac.new(
            self.stripe_webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def process_event(self, event: WebhookEvent) -> dict[str, Any]:
        """
        Process Stripe webhook event with idempotency.

        Args:
            event: Parsed webhook event

        Returns:
            Processing result with status and subscription details
        """
        # Idempotency check
        if event.event_id in self._processed_events:
            return {
                "status": "already_processed",
                "event_id": event.event_id,
                "event_type": event.event_type
            }

        # Route to handler
        handler_map = {
            "checkout.session.completed": self._handle_checkout_completed,
            "invoice.payment_succeeded": self._handle_payment_succeeded,
            "invoice.payment_failed": self._handle_payment_failed,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
        }

        handler = handler_map.get(event.event_type)
        if not handler:
            return {
                "status": "unhandled_event_type",
                "event_id": event.event_id,
                "event_type": event.event_type
            }

        try:
            result = handler(event.data)
            self._mark_processed(event.event_id, event.event_type, "success")
            return {
                "status": "processed",
                "event_id": event.event_id,
                "event_type": event.event_type,
                "result": result
            }
        except Exception as e:
            self._mark_processed(event.event_id, event.event_type, f"error:{str(e)}")
            return {
                "status": "error",
                "event_id": event.event_id,
                "event_type": event.event_type,
                "error": str(e)
            }

    def _handle_checkout_completed(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle successful checkout session (new subscription)."""
        session = data.get("object", {})
        customer_id = session.get("customer")
        subscription_id_stripe = session.get("subscription")
        metadata = session.get("metadata", {})
        tier_str = metadata.get("tier", "starter")

        tier = SubscriptionTier(tier_str)
        subscription, receipt = self.subscription_service.create(
            customer_id=customer_id,
            tier=tier,
            stripe_subscription_id=subscription_id_stripe,
            trial_days=0
        )

        return {
            "subscription_id": subscription.subscription_id,
            "customer_id": customer_id,
            "tier": tier.value,
            "receipt_id": receipt.receipt_id
        }

    def _handle_payment_succeeded(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle successful invoice payment (renewal)."""
        invoice = data.get("object", {})
        subscription_id_stripe = invoice.get("subscription")

        # Find subscription by Stripe ID
        subscription = self._find_by_stripe_id(subscription_id_stripe)
        if not subscription:
            raise ValueError(f"Subscription not found for Stripe ID: {subscription_id_stripe}")

        subscription, receipt = self.subscription_service.renew(
            subscription_id=subscription.subscription_id,
            payment_succeeded=True
        )

        return {
            "subscription_id": subscription.subscription_id,
            "status": "renewed",
            "receipt_id": receipt.receipt_id
        }

    def _handle_payment_failed(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle failed invoice payment."""
        invoice = data.get("object", {})
        subscription_id_stripe = invoice.get("subscription")

        subscription = self._find_by_stripe_id(subscription_id_stripe)
        if not subscription:
            raise ValueError(f"Subscription not found for Stripe ID: {subscription_id_stripe}")

        subscription, receipt = self.subscription_service.renew(
            subscription_id=subscription.subscription_id,
            payment_succeeded=False
        )

        return {
            "subscription_id": subscription.subscription_id,
            "status": "past_due",
            "receipt_id": receipt.receipt_id
        }

    def _handle_subscription_updated(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle subscription tier change."""
        stripe_subscription = data.get("object", {})
        subscription_id_stripe = stripe_subscription.get("id")
        metadata = stripe_subscription.get("metadata", {})
        new_tier_str = metadata.get("tier")

        if not new_tier_str:
            return {"status": "no_tier_change"}

        subscription = self._find_by_stripe_id(subscription_id_stripe)
        if not subscription:
            raise ValueError(f"Subscription not found for Stripe ID: {subscription_id_stripe}")

        new_tier = SubscriptionTier(new_tier_str)

        # Determine upgrade vs downgrade
        tier_order = [SubscriptionTier.STARTER, SubscriptionTier.PROFESSIONAL, SubscriptionTier.ENTERPRISE]
        if tier_order.index(new_tier) > tier_order.index(subscription.tier):
            subscription, receipt = self.subscription_service.upgrade(
                subscription_id=subscription.subscription_id,
                new_tier=new_tier
            )
            action = "upgraded"
        else:
            subscription, receipt = self.subscription_service.downgrade(
                subscription_id=subscription.subscription_id,
                new_tier=new_tier
            )
            action = "downgraded"

        return {
            "subscription_id": subscription.subscription_id,
            "action": action,
            "new_tier": new_tier.value,
            "receipt_id": receipt.receipt_id
        }

    def _handle_subscription_deleted(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle subscription cancellation."""
        stripe_subscription = data.get("object", {})
        subscription_id_stripe = stripe_subscription.get("id")
        cancellation_details = stripe_subscription.get("cancellation_details", {})
        reason = cancellation_details.get("reason", "customer_request")

        subscription = self._find_by_stripe_id(subscription_id_stripe)
        if not subscription:
            raise ValueError(f"Subscription not found for Stripe ID: {subscription_id_stripe}")

        subscription, receipt = self.subscription_service.cancel(
            subscription_id=subscription.subscription_id,
            reason=reason
        )

        return {
            "subscription_id": subscription.subscription_id,
            "status": "canceled",
            "reason": reason,
            "receipt_id": receipt.receipt_id
        }

    def _find_by_stripe_id(self, stripe_subscription_id: str):
        """Find subscription by Stripe subscription ID."""
        for sub in self.subscription_service.list_subscriptions():
            if sub.stripe_subscription_id == stripe_subscription_id:
                return sub
        return None


def _self_test() -> bool:
    """
    Runtime verification test for WebhookService (AEP-001/RVC-001/POST-AUDIT-001).

    Tests webhook processing, idempotency, and signature verification.
    """
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_ledger = Path(f.name)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_events = Path(f.name)

    try:
        # Setup
        sub_service = SubscriptionService(ledger_path=test_ledger)
        webhook_service = WebhookService(
            subscription_service=sub_service,
            processed_events_path=test_events
        )

        # Test 1: Checkout completed (new subscription)
        checkout_event = WebhookEvent(
            event_id="evt_test_001",
            event_type="checkout.session.completed",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "customer": "cus_test_001",
                    "subscription": "sub_stripe_test_001",
                    "metadata": {"tier": "professional"}
                }
            }
        )

        result = webhook_service.process_event(checkout_event)
        assert result["status"] == "processed"
        assert "subscription_id" in result["result"]

        # Test 2: Idempotency - same event again
        result_duplicate = webhook_service.process_event(checkout_event)
        assert result_duplicate["status"] == "already_processed"

        # Test 3: Payment succeeded (renewal)
        subscription = list(sub_service.list_subscriptions())[0]
        payment_event = WebhookEvent(
            event_id="evt_test_002",
            event_type="invoice.payment_succeeded",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "subscription": "sub_stripe_test_001"
                }
            }
        )

        result_payment = webhook_service.process_event(payment_event)
        assert result_payment["status"] == "processed"
        assert result_payment["result"]["status"] == "renewed"

        # Test 4: Subscription deleted (cancellation)
        cancel_event = WebhookEvent(
            event_id="evt_test_003",
            event_type="customer.subscription.deleted",
            created=int(datetime.now(timezone.utc).timestamp()),
            livemode=False,
            data={
                "object": {
                    "id": "sub_stripe_test_001",
                    "cancellation_details": {"reason": "customer_request"}
                }
            }
        )

        result_cancel = webhook_service.process_event(cancel_event)
        assert result_cancel["status"] == "processed"
        assert result_cancel["result"]["status"] == "canceled"

        # Test 5: Signature verification
        test_payload = b'{"event": "test"}'
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        signed_payload = f"{timestamp}.{test_payload.decode()}"
        signature = hmac.new(
            webhook_service.stripe_webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256
        ).hexdigest()
        signature_header = f"t={timestamp},v1={signature}"

        is_valid = webhook_service.verify_signature(test_payload, signature_header)
        assert is_valid is True

        # Test 6: Invalid signature
        invalid_sig = webhook_service.verify_signature(test_payload, "t=123,v1=invalid")
        assert invalid_sig is False

        print(f"✓ WebhookService self-test passed (6 scenarios validated, idempotency enforced)")
        return True

    finally:
        test_ledger.unlink(missing_ok=True)
        test_events.unlink(missing_ok=True)


if __name__ == "__main__":
    assert _self_test(), "WebhookService self-test failed"
