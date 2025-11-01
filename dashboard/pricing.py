"""
Tessrax Pricing Dashboard
Module: tessrax.dashboard.pricing
Governance Clauses: AEP-001, RVC-001

Streamlit UI for displaying pricing tiers and subscription features.
Integrates with monetization API for tier comparison.
"""

import streamlit as st
import requests
from typing import Any


def render_pricing_page(api_base_url: str = "http://localhost:8000") -> None:
    """
    Render pricing tiers comparison page.

    Args:
        api_base_url: Base URL for Tessrax API
    """
    st.title("🎯 Tessrax Pricing")
    st.markdown("""
    **Truth detection powered by governance-first AI**

    All tiers include Merkle-anchored receipts with integrity ≥ 0.94 for audit compliance.
    """)

    # Fetch pricing from API
    try:
        response = requests.get(f"{api_base_url}/billing/pricing", timeout=5)
        response.raise_for_status()
        pricing_data = response.json()
        tiers = pricing_data.get("tiers", {})
    except Exception as e:
        st.error(f"Failed to load pricing: {e}")
        # Fallback to hardcoded pricing
        tiers = {
            "starter": {
                "price_monthly_usd": 49.00,
                "quota_limit": 10_000,
                "features": ["contradiction_detection", "basic_dashboard", "email_support", "7_day_retention"],
                "recommended": False
            },
            "professional": {
                "price_monthly_usd": 249.00,
                "quota_limit": 100_000,
                "features": ["contradiction_detection", "advanced_analytics", "api_access", "90_day_retention", "priority_support"],
                "recommended": True
            },
            "enterprise": {
                "price_monthly_usd": 2499.00,
                "quota_limit": None,
                "features": ["dedicated_kernel", "sla_99_9", "sso", "1_year_retention", "custom_policy_compiler", "white_glove_support"],
                "recommended": False
            }
        }

    # Feature display names
    feature_names = {
        "contradiction_detection": "✓ Contradiction Detection API",
        "basic_dashboard": "✓ Basic Dashboard",
        "advanced_analytics": "✓ Advanced Analytics",
        "api_access": "✓ Full API Access",
        "email_support": "✓ Email Support",
        "priority_support": "✓ Priority Support",
        "white_glove_support": "✓ White Glove Support",
        "7_day_retention": "✓ 7-Day Receipt Retention",
        "90_day_retention": "✓ 90-Day Receipt Retention",
        "1_year_retention": "✓ 1-Year Receipt Retention",
        "dedicated_kernel": "✓ Dedicated Governance Kernel",
        "sla_99_9": "✓ 99.9% SLA",
        "sso": "✓ SSO Integration",
        "custom_policy_compiler": "✓ Custom Policy Compiler",
        "export_receipts": "✓ Receipt Export",
        "audit_logs": "✓ Audit Log Access"
    }

    # Render tier cards
    cols = st.columns(3)

    tier_order = ["starter", "professional", "enterprise"]
    tier_display_names = {
        "starter": "Starter",
        "professional": "Professional",
        "enterprise": "Enterprise"
    }

    for idx, tier_key in enumerate(tier_order):
        tier_config = tiers.get(tier_key, {})
        with cols[idx]:
            # Highlight recommended tier
            if tier_config.get("recommended", False):
                st.markdown("### ⭐ **Recommended**")

            st.markdown(f"## {tier_display_names[tier_key]}")

            # Price
            price = tier_config.get("price_monthly_usd", 0)
            st.markdown(f"### ${price:.0f}/month")

            # Quota
            quota = tier_config.get("quota_limit")
            if quota is None:
                st.markdown("**Unlimited** API calls")
            else:
                st.markdown(f"**{quota:,}** API calls/month")

            # Features
            st.markdown("---")
            features = tier_config.get("features", [])
            for feature_key in features:
                feature_display = feature_names.get(feature_key, f"✓ {feature_key}")
                st.markdown(feature_display)

            # CTA button
            st.markdown("---")
            if st.button(f"Subscribe to {tier_display_names[tier_key]}", key=f"btn_{tier_key}"):
                st.session_state["selected_tier"] = tier_key
                st.session_state["show_subscribe_form"] = True
                st.rerun()

    # Subscription form (if tier selected)
    if st.session_state.get("show_subscribe_form", False):
        st.markdown("---")
        render_subscribe_form(
            api_base_url=api_base_url,
            preselected_tier=st.session_state.get("selected_tier", "starter")
        )


def render_subscribe_form(api_base_url: str, preselected_tier: str) -> None:
    """
    Render subscription creation form.

    Args:
        api_base_url: Base URL for API
        preselected_tier: Pre-selected tier from pricing cards
    """
    st.markdown("## Complete Your Subscription")

    with st.form("subscribe_form"):
        customer_email = st.text_input("Email Address", placeholder="your@email.com")
        tier = st.selectbox(
            "Tier",
            options=["starter", "professional", "enterprise"],
            index=["starter", "professional", "enterprise"].index(preselected_tier)
        )
        trial_days = st.slider("Trial Period (days)", min_value=0, max_value=30, value=14)

        submitted = st.form_submit_button("Create Subscription")

        if submitted:
            if not customer_email:
                st.error("Please enter your email address")
            else:
                # Hash customer email for privacy
                import hashlib
                customer_id = f"sha256:{hashlib.sha256(customer_email.encode()).hexdigest()[:32]}"

                # Call subscription API
                try:
                    response = requests.post(
                        f"{api_base_url}/billing/subscribe",
                        json={
                            "customer_id": customer_id,
                            "tier": tier,
                            "trial_days": trial_days
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.success("✅ Subscription created successfully!")
                    st.json(result)

                    # Store subscription ID in session
                    st.session_state["subscription_id"] = result["subscription_id"]
                    st.session_state["show_subscribe_form"] = False

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to create subscription: {e}")

    if st.button("Cancel"):
        st.session_state["show_subscribe_form"] = False
        st.rerun()


def render_tier_comparison() -> None:
    """Render detailed tier comparison table."""
    st.markdown("## Detailed Feature Comparison")

    comparison_data = {
        "Feature": [
            "Contradiction Detection API",
            "Monthly API Calls",
            "Receipt Retention",
            "Dashboard Access",
            "Analytics",
            "Support Level",
            "SLA",
            "SSO Integration",
            "Dedicated Kernel",
            "Custom Policies",
            "Audit Logs",
            "Receipt Export"
        ],
        "Starter": [
            "✓", "10,000", "7 days", "Basic", "—", "Email", "—", "—", "—", "—", "—", "—"
        ],
        "Professional": [
            "✓", "100,000", "90 days", "Advanced", "✓", "Priority", "—", "—", "—", "—", "—", "✓"
        ],
        "Enterprise": [
            "✓", "Unlimited", "1 year", "Advanced", "✓", "White Glove", "99.9%", "✓", "✓", "✓", "✓", "✓"
        ]
    }

    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.table(df)


if __name__ == "__main__":
    # Initialize session state
    if "show_subscribe_form" not in st.session_state:
        st.session_state["show_subscribe_form"] = False

    render_pricing_page()
    st.markdown("---")
    render_tier_comparison()
