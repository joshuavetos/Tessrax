"""Stripe integration utilities with deterministic test-mode fallbacks."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass

try:  # pragma: no cover - stripe optional in CI
    import stripe
except Exception:  # pragma: no cover - degrade gracefully
    stripe = None  # type: ignore


@dataclass
class CheckoutSession:
    """Abstraction over Stripe checkout session payloads."""

    id: str
    url: str
    amount_total: int
    currency: str


class StripeGateway:
    """Create checkout sessions in Stripe test mode with mock fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        mock: bool | None = None,
    ) -> None:
        env_key = api_key or os.getenv("STRIPE_TEST_SECRET_KEY")
        self.api_key = env_key or "sk_test_mocked_tessrax"
        self.mock = bool(mock) if mock is not None else self.api_key.startswith("sk_test_mock")
        if stripe is not None:
            stripe.api_key = self.api_key
            stripe.api_version = "2022-11-15"

    def create_checkout_session(
        self,
        amount_cents: int,
        currency: str,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> CheckoutSession:
        if amount_cents <= 0:
            raise ValueError("Checkout amount must be positive")
        if self.mock or stripe is None:
            fake_id = "cs_test_" + secrets.token_hex(12)
            fake_url = f"https://dashboard.stripe.com/test/sessions/{fake_id}"
            return CheckoutSession(id=fake_id, url=fake_url, amount_total=amount_cents, currency=currency)
        session = stripe.checkout.Session.create(  # type: ignore[attr-defined]
            mode="subscription",
            line_items=[
                {
                    "price_data": {
                        "currency": currency,
                        "unit_amount": amount_cents,
                        "product_data": {
                            "name": f"Tessrax {tier.title()} Plan",
                        },
                    },
                    "quantity": 1,
                }
            ],
            success_url=success_url,
            cancel_url=cancel_url,
            subscription_data={"metadata": {"tier": tier}},
        )
        return CheckoutSession(
            id=session["id"],
            url=session["url"],
            amount_total=session["amount_total"] if "amount_total" in session else amount_cents,
            currency=session.get("currency", currency),
        )


__all__ = ["StripeGateway", "CheckoutSession"]
