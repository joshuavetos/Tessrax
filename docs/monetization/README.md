# Tessrax Monetization Layer v1 — Governed Revenue Stack

**Version:** 1.0.0
**Status:** Production-Ready
**Governance Clauses:** AEP-001, RVC-001, EAC-001, POST-AUDIT-001, DLK-001

---

## Executive Summary

This monetization layer extends Tessrax with a production-grade subscription billing system anchored to the existing Merkle ledger. Every transaction—from signup to cancellation—emits a cryptographically-signed governance receipt with integrity ≥ 0.94, providing audit-ready proof for revenue recognition and compliance.

**Key Features:**
- **Tiered SaaS Subscriptions:** Starter ($49/mo), Professional ($249/mo), Enterprise ($2,499/mo)
- **Ledger-Anchored Billing:** All transactions written to immutable JSONL ledger with Merkle proofs
- **Stripe Integration:** Payment processing via Stripe Checkout and webhooks
- **Quota Enforcement:** Real-time usage metering with entitlement gates
- **Governance Compliance:** Every receipt satisfies DLK-VERIFIED audit requirements
- **Dashboard UI:** Streamlit-based pricing and subscription management pages
- **OpenAPI Spec:** REST API for programmatic subscription management

**Non-Intrusive Design:** Zero modifications to core Memory/Metabolism/Governance/Trust engines. All revenue logic implemented as extension services.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Product Logic](#product-logic)
3. [Pricing Policy](#pricing-policy)
4. [Governance Mapping](#governance-mapping)
5. [Privacy Model](#privacy-model)
6. [Ledger Anchoring](#ledger-anchoring)
7. [Deployment Guide](#deployment-guide)
8. [API Reference](#api-reference)
9. [Testing & Verification](#testing--verification)
10. [Rollback Controls](#rollback-controls)
11. [Developer Guide](#developer-guide)

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tessrax Truth API                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FastAPI Application (main.py)                           │   │
│  │  ├─ /billing/subscribe                                   │   │
│  │  ├─ /billing/upgrade                                     │   │
│  │  ├─ /billing/manage                                      │   │
│  │  ├─ /billing/usage/{subscription_id}                     │   │
│  │  ├─ /billing/history/{customer_id}                       │   │
│  │  ├─ /billing/entitlement/{subscription_id}               │   │
│  │  ├─ /billing/webhooks/stripe                             │   │
│  │  └─ /billing/pricing                                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Services Layer (tessrax_truth_api/services/)            │   │
│  │  ├─ SubscriptionService (lifecycle management)           │   │
│  │  │  • create, upgrade, downgrade, cancel, renew          │   │
│  │  │  • record_usage, get_subscription                     │   │
│  │  ├─ EntitlementService (RBAC & quota gates)              │   │
│  │  │  • check_entitlement, has_feature                     │   │
│  │  ├─ WebhookService (Stripe event processing)             │   │
│  │  │  • process_event, verify_signature                    │   │
│  │  └─ BillingService (existing usage tracker)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Ledger Integration (no core modification)               │   │
│  │  ├─ billing_ledger.jsonl (subscription events)           │   │
│  │  ├─ webhook_events.jsonl (idempotency log)               │   │
│  │  ├─ MerkleEngine (receipt anchoring)                     │   │
│  │  └─ GovernanceKernel (compliance validation)             │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐         ┌──────────────────┐
                  │  Stripe API     │         │  Dashboard UI    │
                  │  • Checkout     │         │  • Pricing Page  │
                  │  • Subscriptions│         │  • Usage Tracker │
                  │  • Webhooks     │         │  • Billing Hist  │
                  └─────────────────┘         └──────────────────┘
```

### Data Flow

1. **Subscription Creation:**
   - User selects tier on pricing page → POST `/billing/subscribe`
   - `SubscriptionService.create()` generates subscription record
   - Governance receipt emitted with `integrity_score ≥ 0.94`
   - Receipt appended to `billing_ledger.jsonl`
   - Stripe checkout session created for payment

2. **Usage Metering:**
   - Each `/detect` API call → `EntitlementService.check_entitlement()`
   - Quota verified → `SubscriptionService.record_usage()`
   - Usage count incremented in-memory and persisted to ledger on state change

3. **Webhook Processing:**
   - Stripe sends `invoice.payment_succeeded` → POST `/billing/webhooks/stripe`
   - `WebhookService.verify_signature()` validates authenticity
   - Event processed via `process_event()` → subscription renewed
   - Governance receipt emitted to ledger

4. **Ledger Anchoring:**
   - Hourly cron job runs `MerkleEngine.build_and_store()`
   - All billing receipts hashed into Merkle tree
   - Root hash computed and stored in `audit_receipt` field
   - Proofs generated for independent verification

---

## Product Logic

### Subscription Tiers

| Tier | Price | Quota | Key Features |
|------|-------|-------|--------------|
| **Starter** | $49/month | 10,000 API calls | Basic dashboard, 7-day retention, email support |
| **Professional** | $249/month | 100,000 API calls | Advanced analytics, 90-day retention, API access, priority support |
| **Enterprise** | $2,499/month | Unlimited | Dedicated kernel, 1-year retention, SLA, SSO, custom policies |

### Lifecycle States

```
TRIALING → ACTIVE → PAST_DUE → CANCELED
    ↓         ↓         ↓
  (trial)  (renew)  (retry)
```

**State Transitions:**
- `TRIALING`: New subscription with trial period (0-30 days)
- `ACTIVE`: Paid subscription in good standing
- `PAST_DUE`: Payment failed, retry scheduled (grace period: 7 days)
- `CANCELED`: User-initiated or payment failure after retries
- `UNPAID`: Hard suspension after grace period

### Upgrade/Downgrade Rules

- **Upgrades:** Immediate effect, prorated billing
- **Downgrades:** Effective at next billing period (no prorating)
- **Quota Reset:** Usage resets to 0 on tier change
- **Feature Access:** New entitlements granted immediately on upgrade

---

## Pricing Policy

### Billing Cycle
- **Monthly recurring:** Charges on same day each month
- **Annual prepay:** 15% discount (future enhancement)
- **Trial period:** 0-30 days configurable per signup

### Payment Processing
- **Provider:** Stripe (PCI-DSS Level 1 certified)
- **Methods:** Credit card, ACH (US), SEPA (EU)
- **Currency:** USD (multi-currency support planned)
- **Invoicing:** Automated via Stripe with 14-day net terms for Enterprise

### Refund Policy
- **Trial cancellation:** Full refund if canceled within trial
- **Mid-cycle cancel:** No prorating; access until period end
- **Service failure:** Prorated credit for SLA violations (Enterprise only)
- **Dispute resolution:** Governance receipts serve as proof of service delivery

### Late Payment Handling
1. **Day 0:** Payment fails → `PAST_DUE` status, email notification
2. **Day 3:** First retry attempt
3. **Day 7:** Second retry, suspend API access
4. **Day 14:** Final retry, if failed → `UNPAID` status, account locked

---

## Governance Mapping

### Receipt Structure

Every billing transaction emits a governance receipt with these fields:

```json
{
  "receipt_id": "rcpt_{sha256:24}",
  "module": "tessrax.billing.{event_type}",
  "event_type": "subscription.created | payment.succeeded | ...",
  "subscription_id": "f3a2b1c0d9e8f7a6",
  "customer_id": "sha256:d4e5f6g7h8i9j0k1",
  "tier": "professional",
  "amount": 249.00,
  "status": "success | failed",
  "metrics": { "quota_limit": 100000, "usage_count": 0, ... },
  "integrity_score": 0.97,
  "merkle_root": "5f9a8b7c...",
  "merkle_proof": ["L:a3b2...", "R:e4f3..."],
  "governance_lane": "general_lane | review_lane | high_priority_lane",
  "compliance_metadata": {
    "pci_dss_exemption": "no_card_data_stored",
    "gdpr_processing_basis": "contract_performance",
    "data_retention_days": 2555,
    "privacy_mode": "pseudonymized"
  },
  "timestamp": "2025-11-01T10:15:30.123456Z",
  "signature": "d9e8f7a6...",
  "auditor": "Tessrax Governance Kernel v16",
  "clauses": ["AEP-001", "RVC-001", "EAC-001", "POST-AUDIT-001", "DLK-001"]
}
```

### Governance Lane Routing

| Event Type | Lane | Rationale |
|------------|------|-----------|
| `subscription.created` | `general_lane` | Normal business operation |
| `subscription.upgraded` | `general_lane` | Customer-initiated change |
| `payment.succeeded` | `general_lane` | Successful transaction |
| `payment.failed` | `review_lane` | Requires manual review |
| `subscription.canceled` | `general_lane` | Normal churn unless disputed |
| `refund.issued` | `high_priority_lane` | Financial reversal, fraud check |

### Compliance Hooks

1. **AEP-001 (Auto-Executable Protocol):** All pricing logic is deterministic and cold-start compatible.
2. **RVC-001 (Runtime Verified Computation):** Every subscription state change verified via `_self_test()` in CI.
3. **EAC-001 (Evidence Aligned Computation):** Stripe webhook events serve as external evidence for payment state.
4. **POST-AUDIT-001 (Post-Audit Compliance):** All receipts written to append-only ledger with Merkle proofs.
5. **DLK-001 (Double-Lock Verification):** Receipts include both HMAC signature and Merkle proof for tamper detection.

---

## Privacy Model

### Data Minimization

**Stored in Tessrax:**
- `customer_id`: SHA256 hash of email (pseudonymized)
- `subscription_id`: Deterministic hash (non-reversible)
- `tier`: Subscription tier (public information)
- `usage_count`: Aggregate API calls (no payload data)

**NOT Stored in Tessrax:**
- Credit card numbers (stored only in Stripe)
- Full email addresses (hashed before storage)
- PII beyond transaction metadata
- API request payloads (only contradiction scores stored)

### Encryption

- **At Rest:** Billing ledger encrypted with AES-256-GCM, keys rotated quarterly
- **In Transit:** All Stripe API calls use TLS 1.3
- **Webhook Signatures:** HMAC-SHA256 validated on all incoming events

### GDPR Compliance

**Legal Basis:** Contract performance (GDPR Article 6.1(b))

**Data Subject Rights:**
- **Right to Access:** `/billing/history/{customer_id}` endpoint returns all receipts
- **Right to Erasure:** Ledger entries pseudonymized; Stripe holds raw PII and handles deletion requests
- **Right to Portability:** Receipts exportable as JSON with Merkle proofs
- **Right to Object:** Customers can cancel subscription at any time

**Data Retention:**
- **Billing receipts:** 7 years (tax compliance requirement)
- **Compressed summaries:** Transferred to Memory engine after 90 days
- **Stripe data:** Per Stripe's data retention policies

---

## Ledger Anchoring

### How Anchoring Works

1. **Receipt Generation:** Each billing event writes a JSON receipt to `ledger/billing_ledger.jsonl`
2. **Batch Processing:** Hourly cron job collects all new receipts
3. **Merkle Tree Construction:**
   - Hash each receipt: `SHA256(canonical_json(receipt))`
   - Build binary tree from hashes
   - Compute root: `merkle_root = SHA256(left_hash + right_hash)`
4. **Proof Generation:** For each receipt, generate sibling proof path (`L:hash` or `R:hash`)
5. **Receipt Annotation:** Update each receipt with `merkle_root` and `merkle_proof` fields
6. **External Anchoring (Optional):** Write `merkle_root` to blockchain for immutability

### Verification Process

**Independent Auditor Steps:**

1. Load receipt from ledger:
   ```bash
   cat ledger/billing_ledger.jsonl | grep "rcpt_a1b2c3d4e5f6g7h8"
   ```

2. Extract `merkle_root` and `merkle_proof` from receipt

3. Recompute leaf hash:
   ```python
   import hashlib, json
   leaf_hash = hashlib.sha256(
       json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
   ).hexdigest()
   ```

4. Verify proof:
   ```python
   from tessrax.core.merkle_engine import MerkleEngine
   is_valid = MerkleEngine.verify_merkle_proof(
       leaf_hash, proof, merkle_root
   )
   assert is_valid, "Receipt tampered"
   ```

**Result:** Cryptographic proof that receipt existed at time of Merkle tree construction and has not been altered.

---

## Deployment Guide

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ or Redis (for production subscription cache)
- Stripe account with API keys
- Kubernetes 1.24+ or Docker Compose

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment variables
export STRIPE_SECRET_KEY=sk_test_...
export STRIPE_WEBHOOK_SECRET=whsec_...
export BILLING_LEDGER_PATH=ledger/billing_ledger.jsonl
export HMAC_SECRET=your-hmac-secret

# 3. Initialize database (optional for persistent cache)
# psql -U postgres -c "CREATE DATABASE tessrax_billing;"

# 4. Run migrations (if using DB)
# alembic upgrade head

# 5. Start API server
uvicorn tessrax_truth_api.main:app --host 0.0.0.0 --port 8000
```

### Kubernetes Deployment

```yaml
# k8s/billing-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tessrax-billing
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tessrax-billing
  template:
    metadata:
      labels:
        app: tessrax-billing
    spec:
      containers:
      - name: api
        image: tessrax/truth-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: STRIPE_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: stripe-secrets
              key: secret-key
        - name: BILLING_LEDGER_PATH
          value: /mnt/ledger/billing_ledger.jsonl
        volumeMounts:
        - name: ledger-volume
          mountPath: /mnt/ledger
      volumes:
      - name: ledger-volume
        persistentVolumeClaim:
          claimName: billing-ledger-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: tessrax-billing-svc
spec:
  selector:
    app: tessrax-billing
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Stripe Webhook Configuration

1. Log in to Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://api.tessrax.com/billing/webhooks/stripe`
3. Select events:
   - `checkout.session.completed`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy webhook signing secret to `STRIPE_WEBHOOK_SECRET`

### Feature Flags

```yaml
# config/features.yml
monetization:
  enabled: true
  tiers:
    starter:
      enabled: true
    professional:
      enabled: true
    enterprise:
      enabled: false  # Beta release, enable after testing

  stripe_integration:
    enabled: true
    test_mode: false  # Set to true for staging

  ledger_anchoring:
    enabled: true
    batch_interval_minutes: 60
```

### Monitoring

```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Sample metrics:
# billing_subscriptions_active{tier="professional"} 123
# billing_revenue_mrr_usd{tier="professional"} 30627.00
# billing_quota_usage_pct{subscription_id="f3a2b1c0"} 0.67
# billing_webhook_events_total{event_type="payment.succeeded"} 456
```

---

## API Reference

See [OpenAPI Specification](./openapi_spec.yaml) for complete API documentation.

### Quick Reference

**Create Subscription:**
```bash
curl -X POST https://api.tessrax.com/billing/subscribe \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "sha256:a1b2c3d4...",
    "tier": "professional",
    "trial_days": 14
  }'
```

**Check Usage:**
```bash
curl https://api.tessrax.com/billing/usage/f3a2b1c0d9e8f7a6 \
  -H "Authorization: Bearer {jwt_token}"
```

**Upgrade Tier:**
```bash
curl -X POST https://api.tessrax.com/billing/upgrade \
  -H "Authorization: Bearer {jwt_token}" \
  -d '{
    "subscription_id": "f3a2b1c0d9e8f7a6",
    "new_tier": "enterprise"
  }'
```

---

## Testing & Verification

### Run Test Suite

```bash
# All monetization tests
pytest tests/test_monetization.py -v

# Specific test class
pytest tests/test_monetization.py::TestSubscriptionLifecycle -v

# With coverage
pytest tests/test_monetization.py --cov=tessrax_truth_api/services --cov-report=html
```

### Self-Test Validation

```bash
# Run service self-tests
python -m tessrax_truth_api.services.subscription_service
python -m tessrax_truth_api.services.entitlement_service
python -m tessrax_truth_api.services.webhook_service

# Expected output:
# ✓ SubscriptionService self-test passed (5 lifecycle events, all receipts integrity ≥ 0.94)
# ✓ EntitlementService self-test passed (7 scenarios validated)
# ✓ WebhookService self-test passed (6 scenarios validated, idempotency enforced)
```

### Ledger Verification

```bash
# Verify all receipts have valid Merkle proofs
python -m tessrax.core.merkle_engine verify ledger/billing_ledger.jsonl

# Check integrity scores
jq '.integrity_score' ledger/billing_ledger.jsonl | awk '{if($1<0.94) print "FAIL: "$1; else print "PASS"}'
```

---

## Rollback Controls

### Rollback Procedure

If critical issues are detected post-deployment:

```bash
# 1. Disable new subscriptions
kubectl set env deployment/tessrax-billing MONETIZATION_ENABLED=false

# 2. Pause webhook processing
kubectl scale deployment/webhook-processor --replicas=0

# 3. Restore previous version
kubectl rollout undo deployment/tessrax-billing

# 4. Verify core engines unaffected
pytest tests/test_governance.py tests/test_ledger.py -v

# 5. Analyze billing ledger for consistency
python scripts/audit_billing_ledger.py --since "2025-11-01T00:00:00Z"
```

### Data Integrity Checks

```python
# scripts/audit_billing_ledger.py
from pathlib import Path
import json

ledger_path = Path("ledger/billing_ledger.jsonl")
issues = []

with open(ledger_path, "r") as f:
    for line_num, line in enumerate(f, 1):
        receipt = json.loads(line)

        # Check integrity score
        if receipt["integrity_score"] < 0.94:
            issues.append(f"Line {line_num}: Low integrity {receipt['integrity_score']}")

        # Check required fields
        required = ["receipt_id", "subscription_id", "merkle_root", "signature"]
        if not all(k in receipt for k in required):
            issues.append(f"Line {line_num}: Missing required fields")

if issues:
    print(f"❌ Found {len(issues)} issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Billing ledger integrity verified")
```

### Graceful Degradation

If Stripe is unavailable:
1. Queue webhook events in Redis/RabbitMQ
2. Continue accepting API requests (cached entitlements)
3. Defer billing state changes until Stripe recovers
4. Emit governance receipts with `status: deferred`

---

## Developer Guide

### Extension Points

**Add New Billing Event:**

1. Define event type in `SubscriptionService`:
   ```python
   def custom_event(self, subscription_id: str, metadata: dict) -> BillingReceipt:
       receipt = self._write_receipt(
           event_type="custom.event_name",
           subscription=subscription,
           compliance_metadata=metadata
       )
       return receipt
   ```

2. Add governance lane routing in `_determine_governance_lane()`:
   ```python
   if event_type == "custom.event_name":
       return "review_lane"
   ```

3. Create test case:
   ```python
   def test_custom_event(self, subscription_service):
       sub, _ = subscription_service.create(...)
       receipt = subscription_service.custom_event(sub.subscription_id, {...})
       assert receipt.integrity_score >= 0.94
   ```

**Add New Tier:**

1. Update `TIER_PRICING` in `SubscriptionService`:
   ```python
   SubscriptionTier.PREMIUM: {
       "price": 999.00,
       "quota": 500_000,
       "features": ["feature_1", "feature_2"]
   }
   ```

2. Add tier to `EntitlementService.TIER_FEATURES`
3. Update OpenAPI spec with new tier enum
4. Add pricing card to `dashboard/pricing.py`

### Audit Hooks

All billing operations call `GovernanceKernel.ingest_record()` for integration with existing audit pipeline:

```python
# In SubscriptionService._write_receipt()
from tessrax.governance import GovernanceKernel

kernel = GovernanceKernel()
lane = kernel.ingest_record({
    "id": receipt.receipt_id,
    "module": receipt.module,
    "event_type": receipt.event_type,
    "integrity": receipt.integrity_score
})
```

### Maintainer Checklist

- [ ] All new services implement `_self_test()` function
- [ ] Tests emit receipts with `integrity_score ≥ 0.94`
- [ ] Ledger writes are atomic (JSONL append-only)
- [ ] Stripe webhooks enforce idempotency
- [ ] PII is hashed before storage
- [ ] Compliance metadata includes GDPR processing basis
- [ ] Merkle proofs generated for all receipts
- [ ] Documentation updated with new features

---

## Changelog

### v1.0.0 (2025-11-01)

**Added:**
- SubscriptionService with create/upgrade/downgrade/cancel/renew
- EntitlementService with quota enforcement and feature gates
- WebhookService with Stripe event processing
- OpenAPI specification for billing endpoints
- Streamlit pricing and subscription management UI
- Sample ledger receipts with Merkle proofs
- Comprehensive test suite (100+ assertions)
- Deployment guides for Kubernetes and Docker Compose

**Governance:**
- All receipts satisfy AEP-001, RVC-001, EAC-001, POST-AUDIT-001, DLK-001
- Integrity scores: 0.94-0.99 range
- GDPR-compliant data minimization and retention policies

**Zero Core Modifications:**
- No changes to Memory/Metabolism/Governance/Trust engines
- Extension-only architecture via service layer

---

## Support & Feedback

- **Documentation:** https://docs.tessrax.com/monetization
- **API Status:** https://status.tessrax.com
- **Support Email:** billing-support@tessrax.com
- **Governance Issues:** File ticket at https://github.com/tessrax/governance/issues

---

## License

Proprietary - Tessrax Platform
© 2025 Tessrax. All rights reserved.
