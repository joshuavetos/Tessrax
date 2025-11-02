# Tessrax Monetization Model Specification
**Model:** Tiered SaaS Subscriptions with Ledger-Anchored Billing
**Document ID:** RVC-MON-002
**Version:** 1.0
**Governance Clauses:** AEP-001, RVC-001, EAC-001, POST-AUDIT-001, DLK-001

---

## Selection Rationale

Tiered SaaS subscriptions are chosen as the MVP monetization model because:

1. **Architectural Fit:** Tessrax already implements JWT-based tier enforcement (`free`, `pro`, `enterprise`) in `tessrax_truth_api/services/billing_service.py`, requiring only subscription lifecycle additions rather than a complete rebuild.

2. **Ledger Synergy:** Every subscription state change (signup, upgrade, cancellation, renewal) can emit a governance receipt to the existing Merkle ledger, providing cryptographic proof for revenue recognition and dispute resolution.

3. **Compliance by Design:** The Governance engine's DLK-VERIFIED receipt pattern satisfies SOC 2 Type II audit requirements for transaction immutability, eliminating the need for separate billing audit infrastructure.

4. **Low Engineering Risk:** Extends existing `BillingService`, `ProvenanceService`, and JWT middleware without modifying core Memory/Metabolism/Governance/Trust engine code, meeting the immutability constraint.

---

## MVP Scope (4-Week Rollout)

### Week 1: Subscription Lifecycle Service
- Implement `SubscriptionService` class with methods: `create`, `upgrade`, `downgrade`, `cancel`, `renew`
- Integrate with Stripe Checkout for payment processing
- Write subscription state changes to dedicated `billing_ledger.jsonl`
- Link each subscription event to Truth API receipts via `receipt_uuid`

### Week 2: Entitlement & Usage Metering
- Extend `BillingService.record_usage()` to deduct from subscription quota
- Add `EntitlementService` to gate `/detect` endpoint on active subscription status
- Implement real-time usage dashboards in Streamlit showing quota consumption
- Create webhook handlers for Stripe events (payment succeeded, failed, subscription updated)

### Week 3: API & Dashboard UI
- Build OpenAPI-compliant REST endpoints: `/subscribe`, `/manage_subscription`, `/usage`, `/billing_history`
- Create pricing page component (`dashboard/pricing.py`) with tier comparison table
- Add subscription management UI (`dashboard/subscription_manager.py`) for upgrades/cancellations
- Implement ledger receipt viewer filtered by `billing_transaction` type

### Week 4: Testing & Documentation
- Write deterministic tests for all subscription lifecycle transitions
- Generate sample ledger receipts for signup, payment, upgrade, refund, cancellation
- Create deployment guide with Kubernetes manifests and Stripe configuration steps
- Document RBAC integration with Governance engine for enterprise SSO

---

## Success Metrics (90-Day Horizon)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Trial-to-Paid Conversion** | 15% | Count Stripe `subscription.created` events with `trial_end` in past |
| **Monthly Recurring Revenue (MRR)** | $10,000 | Sum `amount` from active subscriptions in billing ledger |
| **Ledger Integrity** | 100% | All billing receipts must have `integrity_score ≥ 0.94` and valid Merkle proof |
| **Payment Failure Recovery** | 70% | Ratio of `invoice.payment_succeeded` after `invoice.payment_failed` within 7 days |
| **Churn Rate** | <5%/month | Count `subscription.deleted` events divided by total active subscriptions |

---

## Risk Controls

### Financial Risks
- **Chargeback Fraud:** Stripe Radar enabled for automatic dispute detection; disputed transactions flagged in ledger with `compliance.dispute_status` field.
- **Revenue Leakage:** Hourly cron job reconciles Stripe subscription states with internal billing ledger; discrepancies emit high-priority governance alerts.
- **Price Manipulation:** Pricing config stored in immutable ledger; any tier price change requires governance receipt with `action: PRICING_UPDATE` signed by authorized admin.

### Technical Risks
- **Ledger Corruption:** All billing writes use `MerkleEngine.build_and_store()` with atomic JSONL appends; receipt verification integrated into CI/CD governance workflow.
- **Entitlement Bypass:** JWT claims include `subscription_id` and `tier`; middleware validates against live Stripe subscription status on every request to `/detect`.
- **Webhook Replay Attacks:** Stripe webhook signatures verified using `stripe.Webhook.construct_event()`; each event ID logged to prevent duplicate processing.

### Compliance Risks
- **GDPR Right to Erasure:** Billing ledger implements pseudonymization via hashed `customer_id`; raw PII stored only in Stripe with data retention policies configured.
- **PCI-DSS Scope:** No credit card data stored in Tessrax; all payment processing delegated to Stripe's PCI-compliant infrastructure.
- **Audit Trail Gaps:** Every subscription event includes `merkle_proof` field linking to governance ledger; independent auditors can verify transaction integrity without accessing production database.

---

## Rollout Plan

### Phase 1: Internal Beta (Week 5)
- Deploy to staging environment with test Stripe keys
- Invite 10 internal users to test signup flow and quota enforcement
- Verify all receipts written to ledger have integrity ≥ 0.94

### Phase 2: Limited Public Release (Week 6-8)
- Enable production Stripe integration with real payment processing
- Invite 50 waitlist users to starter tier ($49/month)
- Monitor webhook delivery latency and ledger write throughput
- Publish pricing page to public documentation site

### Phase 3: General Availability (Week 9+)
- Open self-service signup to all users
- Launch enterprise sales motion with dedicated governance kernel upsell
- Implement usage-based overage billing for professional tier
- Add annual prepayment option with 15% discount

---

## Integration with Governance Hooks

Every subscription lifecycle event satisfies governance requirements:

| Event | Governance Lane | Receipt Module | Compliance Clause |
|-------|-----------------|----------------|-------------------|
| **Signup** | `general_lane` | `tessrax.billing.subscription.create` | AEP-001 (deterministic pricing) |
| **Payment Success** | `general_lane` | `tessrax.billing.payment.succeeded` | DLK-001 (double-lock verification) |
| **Payment Failure** | `review_lane` | `tessrax.billing.payment.failed` | POST-AUDIT-001 (retry logic audit trail) |
| **Upgrade** | `general_lane` | `tessrax.billing.subscription.upgrade` | EAC-001 (evidence = new tier entitlement) |
| **Cancellation** | `review_lane` | `tessrax.billing.subscription.cancel` | RVC-001 (runtime quota zeroing verification) |
| **Refund** | `high_priority_lane` | `tessrax.billing.refund.issued` | DLK-001 (fraud detection double-lock) |

Each receipt includes:
- `merkle_root`: Anchors transaction to governance ledger
- `governance_decision`: Links to GovernanceKernel verdict (ACKNOWLEDGE, REVIEW, ESCALATE)
- `trust_score`: BayesianTrust score at time of transaction for fraud analysis
- `compliance_metadata`: PCI-DSS exemption assertion, GDPR processing basis, data retention policy

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tessrax Truth API                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI App (main.py)                               │   │
│  │  ├─ /detect (usage metered)                          │   │
│  │  ├─ /subscribe (Stripe Checkout redirect)            │   │
│  │  ├─ /webhooks/stripe (payment events)                │   │
│  │  ├─ /usage (quota dashboard)                         │   │
│  │  └─ /manage_subscription (upgrade/cancel)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Services Layer                                       │   │
│  │  ├─ SubscriptionService (lifecycle management)       │   │
│  │  ├─ BillingService (quota enforcement)               │   │
│  │  ├─ EntitlementService (RBAC gates)                  │   │
│  │  ├─ ProvenanceService (Merkle anchoring)             │   │
│  │  └─ WebhookService (Stripe event processing)         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Ledger Integration                                   │   │
│  │  ├─ billing_ledger.jsonl (subscription events)       │   │
│  │  ├─ MerkleEngine (receipt anchoring)                 │   │
│  │  └─ GovernanceKernel (compliance validation)         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Stripe API     │
                  │  - Checkout     │
                  │  - Subscriptions│
                  │  - Webhooks     │
                  └─────────────────┘
```

---

## Privacy Model

- **Data Minimization:** Only `customer_id` (hashed), `tier`, `usage_count`, `subscription_status` stored in ledger.
- **Encryption at Rest:** Billing ledger encrypted using AES-256-GCM; keys rotated quarterly.
- **Encryption in Transit:** All Stripe API calls use TLS 1.3; webhook signatures validate payload integrity.
- **Retention Policy:** Raw billing receipts retained 7 years for tax compliance; compressed into Memory engine temporal summaries after 90 days.
- **Right to Access:** Customer can download all billing receipts via `/export_receipts` endpoint; includes Merkle proof for authenticity verification.

---

**Document Status:** Ready for implementation. Proceed to code artifact generation.

**Self-Test Requirement:** This spec must be validated by implementing at least one deterministic test case for each subscription lifecycle state transition, with all tests emitting governance receipts with integrity ≥ 0.94.
