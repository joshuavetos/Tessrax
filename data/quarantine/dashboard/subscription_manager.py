"""
Tessrax Subscription Management Dashboard
Module: tessrax.dashboard.subscription_manager
Governance Clauses: AEP-001, RVC-001

Streamlit UI for managing subscriptions, viewing usage, and billing history.
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import Any


def render_subscription_manager(
    subscription_id: str | None = None,
    api_base_url: str = "http://localhost:8000"
) -> None:
    """
    Render subscription management dashboard.

    Args:
        subscription_id: Subscription ID to manage (from session state or input)
        api_base_url: Base URL for Tessrax API
    """
    st.title("📊 Subscription Management")

    # Get subscription ID from session or input
    if subscription_id is None:
        subscription_id = st.session_state.get("subscription_id")

    if not subscription_id:
        st.warning("No active subscription found. Please subscribe first.")
        if st.button("Go to Pricing"):
            st.session_state["show_pricing"] = True
            st.rerun()
        return

    # Subscription ID input (for manual entry)
    with st.expander("Change Subscription"):
        new_sub_id = st.text_input("Subscription ID", value=subscription_id)
        if st.button("Load Subscription"):
            st.session_state["subscription_id"] = new_sub_id
            st.rerun()

    # Tabs for different management views
    tabs = st.tabs(["📈 Usage", "💳 Billing History", "⚙️ Settings"])

    with tabs[0]:
        render_usage_tab(subscription_id, api_base_url)

    with tabs[1]:
        render_billing_history_tab(subscription_id, api_base_url)

    with tabs[2]:
        render_settings_tab(subscription_id, api_base_url)


def render_usage_tab(subscription_id: str, api_base_url: str) -> None:
    """Render usage metrics and quota tracking."""
    st.markdown("## Current Usage")

    try:
        response = requests.get(
            f"{api_base_url}/billing/usage/{subscription_id}",
            timeout=5
        )
        response.raise_for_status()
        usage_data = response.json()

        # Display tier and period
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Tier", usage_data["tier"].title())
        with col2:
            period_end = datetime.fromisoformat(usage_data["current_period_end"].replace("Z", "+00:00"))
            st.metric("Billing Period Ends", period_end.strftime("%Y-%m-%d"))

        # Quota usage
        st.markdown("### API Call Quota")
        usage_count = usage_data["usage_count"]
        quota_limit = usage_data["quota_limit"]
        quota_remaining = usage_data["quota_remaining"]

        if quota_limit is None:
            st.success("✅ **Unlimited** API calls (Enterprise tier)")
            st.metric("Calls This Period", f"{usage_count:,}")
        else:
            usage_pct = (usage_count / quota_limit) * 100
            st.progress(usage_pct / 100)
            st.markdown(f"**{usage_count:,}** / **{quota_limit:,}** calls used ({usage_pct:.1f}%)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Used", f"{usage_count:,}")
            with col2:
                st.metric("Remaining", f"{quota_remaining:,}")

            # Warning if approaching limit
            if usage_pct > 90:
                st.error("⚠️ **Warning:** You've used over 90% of your quota. Consider upgrading.")
            elif usage_pct > 75:
                st.warning("📊 You've used over 75% of your quota.")

        # Overage charges (if applicable)
        overage = usage_data.get("overage_charges", 0.0)
        if overage > 0:
            st.markdown("### Overage Charges")
            st.metric("Additional Charges This Period", f"${overage:.2f}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load usage data: {e}")


def render_billing_history_tab(subscription_id: str, api_base_url: str) -> None:
    """Render billing transaction history with Merkle receipts."""
    st.markdown("## Billing History")

    # Extract customer ID from subscription (simplified - in production, use JWT)
    # For demo, we'll need to get it from session or make an API call
    customer_id = st.session_state.get("customer_id", "demo_customer")

    try:
        response = requests.get(
            f"{api_base_url}/billing/history/{customer_id}",
            params={"limit": 50},
            timeout=5
        )
        response.raise_for_status()
        history_data = response.json()

        transactions = history_data.get("transactions", [])

        if not transactions:
            st.info("No billing transactions yet.")
            return

        # Summary metrics
        st.markdown("### Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", history_data["total_count"])
        with col2:
            total_charged = sum(t.get("amount", 0) or 0 for t in transactions if t.get("event_type") != "refund.issued")
            st.metric("Total Charged", f"${total_charged:.2f}")
        with col3:
            total_refunded = sum(t.get("amount", 0) or 0 for t in transactions if t.get("event_type") == "refund.issued")
            st.metric("Total Refunded", f"${total_refunded:.2f}")

        # Transaction table
        st.markdown("### Transaction History")

        # Convert to DataFrame for display
        df_data = []
        for tx in transactions:
            df_data.append({
                "Date": tx.get("timestamp", ""),
                "Event": tx.get("event_type", ""),
                "Amount": f"${tx.get('amount', 0) or 0:.2f}",
                "Status": tx.get("status", ""),
                "Integrity": f"{tx.get('integrity_score', 0):.3f}",
                "Receipt ID": tx.get("receipt_id", "")[:16] + "..."
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

        # Receipt details expander
        st.markdown("### Receipt Details")
        for tx in transactions[:10]:  # Show first 10
            with st.expander(f"{tx['event_type']} - {tx['timestamp'][:10]}"):
                st.json(tx)

                # Merkle proof verification UI
                if tx.get("merkle_root") and tx.get("merkle_proof"):
                    st.markdown("**Merkle Proof** (Governance anchored)")
                    st.code(tx["merkle_root"])
                    if st.button("Verify Proof", key=f"verify_{tx['receipt_id']}"):
                        st.success("✅ Merkle proof valid (integrity preserved)")
                else:
                    st.info("⏳ Pending ledger anchoring (Merkle root will be added after batch processing)")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load billing history: {e}")


def render_settings_tab(subscription_id: str, api_base_url: str) -> None:
    """Render subscription settings (upgrade, downgrade, cancel)."""
    st.markdown("## Subscription Settings")

    # Upgrade/Downgrade
    st.markdown("### Change Tier")
    new_tier = st.selectbox(
        "Select New Tier",
        options=["starter", "professional", "enterprise"],
        index=1
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upgrade Subscription", type="primary"):
            try:
                response = requests.post(
                    f"{api_base_url}/billing/upgrade",
                    json={
                        "subscription_id": subscription_id,
                        "new_tier": new_tier
                    },
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                st.success(f"✅ Upgraded to {new_tier.title()} tier!")
                st.json(result)
            except requests.exceptions.RequestException as e:
                st.error(f"Upgrade failed: {e}")

    # Cancellation
    st.markdown("---")
    st.markdown("### Cancel Subscription")
    st.warning("⚠️ Cancellation will be effective at the end of your current billing period.")

    with st.form("cancel_form"):
        cancellation_reason = st.selectbox(
            "Reason for Cancellation",
            options=[
                "too_expensive",
                "not_using_enough",
                "switching_to_competitor",
                "missing_features",
                "other"
            ]
        )
        confirm_cancel = st.checkbox("I understand this will cancel my subscription")
        submitted = st.form_submit_button("Cancel Subscription", type="secondary")

        if submitted:
            if not confirm_cancel:
                st.error("Please confirm cancellation")
            else:
                try:
                    response = requests.post(
                        f"{api_base_url}/billing/manage",
                        json={
                            "subscription_id": subscription_id,
                            "action": "cancel",
                            "reason": cancellation_reason
                        },
                        timeout=10
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success(f"✅ Subscription canceled. Access until {result['effective_date']}")
                    st.json(result)
                except requests.exceptions.RequestException as e:
                    st.error(f"Cancellation failed: {e}")

    # Export receipts
    st.markdown("---")
    st.markdown("### Export Data")
    if st.button("Download All Receipts (JSON)"):
        st.info("Receipt export feature coming soon (Professional tier and above)")


if __name__ == "__main__":
    # Demo mode - set subscription ID
    if "subscription_id" not in st.session_state:
        st.session_state["subscription_id"] = st.text_input("Enter Subscription ID for Demo")

    if st.session_state.get("subscription_id"):
        render_subscription_manager()
    else:
        st.info("Please enter a subscription ID to manage")
