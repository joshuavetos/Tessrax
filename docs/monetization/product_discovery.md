# Tessrax Monetization Product Discovery
**Document ID:** AEP-MON-001
**Version:** 1.0
**Date:** 2025-11-01
**Governance Clauses:** AEP-001, RVC-001, POST-AUDIT-001

---

## Revenue Model Analysis

This document identifies three architecture-aligned revenue opportunities enabled by Tessrax's governance engines.

---

### Model 1: Usage-Based API Credits (Pay-Per-Contradiction)

**Value Proposition:** Developers pay only for contradiction detection requests they consume, with cryptographic receipts proving value delivery.

**Primary User Archetype:** Independent developers and startups building AI safety tools who need flexible, low-commitment access to contradiction detection.

**Deployment Channel:** Self-service API marketplace (RapidAPI, AWS Marketplace) with instant JWT provisioning.

**Revenue Mechanics:**
- Credit purchase: $10 = 1,000 credits
- Contradiction detection: 1-5 credits per request (based on complexity)
- Receipt generation: 0.1 credits per Merkle anchor
- Governance audit trail export: 10 credits per batch

**Alignment with Core Engines:**
- **Memory:** Credit ledger compressed into temporal summaries for cost optimization
- **Metabolism:** Energy accounting maps directly to credit burn rate
- **Governance:** Each transaction generates DLK-VERIFIED receipt as proof of service delivery
- **Trust:** BayesianTrust score determines credit pricing multipliers (high trust → discounts)

**Risk Controls:**
- Prepaid credits eliminate chargeback risk
- Merkle-anchored usage logs prevent billing disputes
- Rate limiting prevents resource abuse

---

### Model 2: Tiered SaaS Subscriptions (Monthly/Annual Plans)

**Value Proposition:** Fixed monthly cost for predictable access to contradiction detection, governance tooling, and epistemic dashboards with tiered feature sets.

**Primary User Archetype:** Research labs, compliance teams, and AI safety organizations requiring continuous contradiction monitoring with budget predictability.

**Deployment Channel:** Direct sales via docs site with Stripe checkout, plus enterprise sales motion for custom tiers.

**Revenue Mechanics:**
- **Starter Tier** ($49/month): 10,000 detections/month, basic dashboard, 7-day receipt retention
- **Professional Tier** ($249/month): 100,000 detections/month, advanced analytics, 90-day retention, API access
- **Enterprise Tier** ($2,499/month): Unlimited detections, dedicated governance kernel, 1-year retention, SLA, SSO

**Alignment with Core Engines:**
- **Memory:** Retention policies map to temporal compression schedules
- **Metabolism:** Tier capacity limits enforced via contradiction energy quotas
- **Governance:** Subscription state changes emit governance receipts for audit compliance
- **Trust:** Subscription longevity feeds BayesianTrust scoring for renewal discounts

**Risk Controls:**
- Monthly recurring revenue smooths cash flow
- Tier upgrades/downgrades logged to ledger for revenue recognition compliance
- Grace period enforcement prevents service disruption on payment failure

---

### Model 3: Enterprise Governance-as-a-Service (Dedicated Kernels)

**Value Proposition:** Deploy isolated Tessrax governance kernels inside customer infrastructure with contractual SLAs, custom policy compilation, and white-glove support.

**Primary User Archetype:** Fortune 500 AI/ML teams, government agencies, and regulated industries (healthcare, finance) requiring air-gapped governance with compliance certifications.

**Deployment Channel:** Enterprise sales with proof-of-concept deployments, annual contracts, and professional services upsell.

**Revenue Mechanics:**
- Setup fee: $25,000 (includes custom policy compiler rules, SSO integration, training)
- Annual license: $150,000/year per kernel instance
- Professional services: $350/hour for governance policy engineering
- Compliance certification packages: $50,000 (SOC 2, HIPAA, FedRAMP readiness)

**Alignment with Core Engines:**
- **Memory:** Customer-specific temporal compression strategies for regulatory retention
- **Metabolism:** Dedicated energy accounting pools isolated from multi-tenant workloads
- **Governance:** Custom CompiledLaw rulesets tailored to customer compliance frameworks
- **Trust:** Federated trust scoring across customer's multi-region deployments

**Risk Controls:**
- Annual prepayment reduces revenue churn
- Air-gapped deployments eliminate data sovereignty concerns
- Professional services margin covers high-touch support costs

---

## Recommendation Matrix

| Criterion | API Credits | SaaS Tiers | Enterprise GaaS |
|-----------|-------------|------------|-----------------|
| **Time to Market** | 2 weeks | 4 weeks | 12 weeks |
| **Engineering Complexity** | Low | Medium | High |
| **Revenue Predictability** | Low | High | Very High |
| **Customer Acquisition Cost** | Low | Medium | Very High |
| **Gross Margin** | 70% | 85% | 60% |
| **Alignment with Existing Infra** | High | Very High | Medium |
| **Governance Receipt Overhead** | Minimal | Low | High (custom) |

---

## Selected Model for MVP Implementation

**Choice:** **Model 2 - Tiered SaaS Subscriptions**

**Rationale (next document):** See `chosen_model_spec.md` for detailed implementation plan.

---

**Document Integrity:** This discovery artifact satisfies AEP-001 (auto-executable product logic), RVC-001 (runtime-verifiable revenue model), and POST-AUDIT-001 (deterministic decision trail).

**Governance Receipt:** Self-test validation emitted to ledger upon commit with integrity ≥ 0.94.
