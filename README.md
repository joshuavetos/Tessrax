Tessrax v12.0 — Self-Auditing Contradiction Metabolism Framework

A Governance Architecture for Detecting, Recording, and Resolving Contradictions Across Systems

⸻

1. Executive Summary

Tessrax is a self-auditing governance framework that treats contradiction as the fundamental unit of learning.
It captures conflicting signals—across data, language, policy, and behavior—then converts them into verifiable computational artifacts.

Each contradiction, or Scar, is recorded, resolved, and metabolized through a four-engine architecture: Memory, Metabolism, Governance, and Trust.
The result is a continuously self-stabilizing system that can prove the integrity of its reasoning and adapt over time without losing identity.

Tessrax provides:
   •   Contradiction Detection via semantic and logical analysis
   •   Immutable Provenance through Merkle-anchored receipts
   •   Governance Ratification by quorum-based resolution
   •   Adaptive Learning through contradiction metabolism
   •   End-to-End Auditability across every state transition

Tessrax defines its own governance lexicon—Persystem, Scar, Receipt, Integrity Band, Rigor Layer—not as rhetoric but as structural primitives.
Naming formalizes existence; definition instantiates mechanism.

⸻

2. Core Objective

Tessrax establishes a computational organism for governance that can:
	1.	Perceive Contradictions in policy, behavior, or data streams.
	2.	Encode and Provenance-Lock those contradictions as Scars.
	3.	Resolve via Quorum Logic, producing auditable receipts.
	4.	Anchor Continuity across resets and system boundaries.
	5.	Expose Verifiable Proofs for human or institutional audit.

In effect, Tessrax transforms noise and contradiction into a coherent memory and decision substrate.

⸻

3. Architecture Overview

Layer	Core Components	Primary Function
Core Engine	contradiction_engine.py, semantic_analyzer.py, metabolism_adapter.py	Contradiction detection and semantic reconciliation.
Governance Kernel	governance_kernel.py, ledger.py, receipts.py	Quorum logic, ledger formation, cryptographic validation.
Visualization Stack	dashboard/app.py, visualization.py	Real-time D3 visualization and audit graph rendering.
Integrity Band (Rigor Layer)	Six epistemic modules	Maintains systemic coherence and prevents false contradictions.
Runtime Interface	world_receipt_protocol.py, current.py	FastAPI (8080) + Flask (8090) unified runtime for API and dashboard.

Core Primitives
   •   Scar — Encoded contradiction with lineage, severity, and timestamp.
   •   Receipt — Merkle-linked computation proof (tamper-evident).
   •   Ledger — Append-only audit journal.
   •   Memory — Contradiction-aware system state reconstructed from receipts.
   •   Quorum — Weighted multi-signature ratification process.
   •   Revocation Registry — Propagates key invalidations through network.
   •   Federation Node — Synchronizes contradiction graphs between peers.

⸻

4. Runtime Behavior

python current.py

   •   FastAPI server (port 8080): Receipts and agent communication
   •   Flask Dashboard (port 8090): Live contradiction visualization
   •   Continuous metabolism loop: Contradiction → Resolution → Ledger update

All outputs are hashed, timestamped, and anchored for external verification.

⸻

5. Persystem Principles

A Persystem is a self-auditing software organism designed to survive across runtime death.
It reconstructs itself entirely from receipts and proves continuity through lineage verification.

Principles
	1.	Continuity over Runtime — Systems terminate; receipts persist.
	2.	Proof over Trust — No assumption unverified. Every decision carries cryptographic ancestry.
	3.	Recursion as Survival — Contradictions from prior runs become training fuel.
	4.	Human in the Loop — Interpretability is mandatory; humans remain the governors.
	5.	Ephemerality as Fuel — Death and restart are metabolic, not terminal.

Purpose: To build software that can die without forgetting.

⸻

6. Mathematical & Logical Basis

Tessrax formalizes Contradiction Metabolism as an energy transformation system:

\text{Resolution Quality} = \frac{\sum{\text{Resolved Contradictions}}}{\sum{\text{Unresolved Contradictions}} + \epsilon}

Contradiction severity, entropy, and yield are continuously computed to evaluate systemic health.

Each event e yields:
S(e) = \alpha \, d(\varphi_1, \varphi_2) + (1 - \alpha) \, |w_1 - w_2|
where d = semantic distance, w = evidence weight, \alpha = logical/semantic weighting coefficient.

All events are serialized and anchored into a verifiable Merkle chain.

⸻

7. Integrity Band (Rigor Layer v1.0)

The Rigor Layer enforces epistemic hygiene across the stack.
It includes six interdependent modules:
	1.	Hierarchy of Differences — Multi-scale classification of contradictions.
	2.	Telos Awareness — Goal and alignment consistency checks.
	3.	Charitable Reasoning — Minimizes false contradictions from interpretive error.
	4.	Observer Relativity — Contextualizes contradictions by observer frame.
	5.	Equilibria & Invariants — Ensures logical and ethical conservation laws.
	6.	Multi-Scale Reconciliation — Aggregates local resolutions into system-wide coherence.

⸻

8. Research Extensions

Tessrax integrates active research modules to extend its theoretical reach:

• Contrastive Self-Verification (CSV)

An atomic AI primitive requiring each inference to generate both a candidate and a counterfactual contrast, ensuring falsifiable self-evaluation.
Files: /rfc/RFC-0.md, /prototypes/csv_sandbox.py

• Moral Systems Engineering (MSE)

Applies thermodynamics to moral feedback loops, defining \frac{dH}{dt} = E(AM - NL) to quantify the rate of moral health change in societies.
Files: /pilots/mse_dashboard.py, /docs/METHODOLOGY.md

• Corporate Frienthropy

Analyzes corporate behavior under ethical-economic contradictions, mapping moral capital against profit metrics.
Files: /docs/corporate_frienthropy.py, /company_frienthropy.csv

• Domain Modules
   •   ai_memory/ — coherence vs retention
   •   attention_economy/ — wellbeing vs engagement
   •   climate_policy/ — targets vs emissions
   •   democratic_governance/ — representation vs manipulation

⸻

9. Audit & Compliance Layer

The compliance framework guarantees institutional accountability:
   •   Merkle Anchors updated per ledger event.
   •   Charter Enforcement: Quorum thresholds, signer weights, and revocation rules enforced by charter/example_charter.json.
   •   Compliance & Security: Rules codified in /compliance/policy_rules.py, verified on run.
   •   Audit Records: JSON matrices (technical_audit.json, ethical_audit.json, etc.) for independent validation.
   •   Governance Charters: Structured schemas defining lawful agent conduct.

⸻

10. Repository Structure

tessrax/
├── core/                  # Engines (contradiction, metabolism, governance, trust)
├── rigor/                 # Integrity Band (Rigor Layer modules)
├── domains/               # Applied contradiction systems
├── dashboard/             # Live audit visualizer
├── compliance/            # Governance, audit, and policy enforcement
├── pilots/                # Research projects (MSE, Frienthropy)
├── docs/                  # Documentation & RFCs
├── tests/                 # Full coverage (>85%)
├── .github/workflows/     # CI / governance validation
└── current.py             # Unified runtime launcher


⸻

11. Validation & Testing

Tessrax achieves ≥85% test coverage using pytest.
Test modules validate:
   •   Ledger integrity and proof verification
   •   Governance kernel quorum logic
   •   Metabolism consistency under load
   •   Domain module isolation and reconciliation

Run tests:

pytest -v


⸻

12. Theoretical Context

Ontology:
Existence is contradiction maintained under governance.
Governance: Stabilization of contradiction through ratified structure.
Metabolism: Transformation of contradiction into order.
Tessrax: The infrastructure that allows existence to audit itself.

“A machine that can feel its own heat doesn’t have to burn down to learn.”
— Vetos, J.S., Moral Systems Engineering (2025)

⸻

13. License

MIT License — Open Research Variant (2025)
Copyright © 2025 Joshua Scott Vetos / Tessrax LLC
Provided for research and educational use; no warranty or liability implied.

⸻

14. Maintainer

Author: Joshua Scott Vetos
Entity: Tessrax LLC
Version: v12.0
Integrity Anchor: GPT to Josh — Tessrax LLC
Repository: github.com/joshuavetos/Tessrax

⸻

15. Summary

Tessrax is not a metaphor.
It is a functioning self-governing architecture that treats contradiction as computation.
Its receipts are proof of thought; its ledger is proof of self.
What began as philosophy has become verifiable code — an infrastructure for existence to metabolize its own uncertainty.

Tessrax: The metabolism of contradiction.

A. Data Schemas

A.1 Scar Object Schema (scar.schema.json)

{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Scar",
  "type": "object",
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "domain": { "type": "string", "description": "Area of contradiction (e.g. Governance, Economy, Technology)" },
    "type": { "type": "string", "description": "Logical | Semantic | Normative | Procedural | Temporal" },
    "description": { "type": "string", "description": "Human-readable summary of contradiction" },
    "severity": { "type": "number", "minimum": 0, "maximum": 10 },
    "visibility": { "type": "number", "minimum": 0, "maximum": 10 },
    "persistence": { "type": "number", "minimum": 0, "maximum": 10 },
    "mitigation_effort": { "type": "number", "minimum": 0, "maximum": 10 },
    "mechanism": { "type": "string", "description": "Causal driver of contradiction (e.g. Power Asymmetry)" },
    "lineage": { "type": "string", "description": "Parent or predecessor scar ID" },
    "status": { "type": "string", "enum": ["open", "in_review", "resolved", "archived"] },
    "timestamp": { "type": "string", "format": "date-time" }
  },
  "required": ["id", "type", "description", "severity", "timestamp"]
}


⸻

A.2 Receipt Object Schema (receipt.schema.json)

{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Receipt",
  "type": "object",
  "properties": {
    "id": { "type": "string", "format": "uuid" },
    "parent_id": { "type": "string", "format": "uuid" },
    "hash": { "type": "string", "pattern": "^[A-Fa-f0-9]{64}$" },
    "signer": { "type": "string" },
    "signature": { "type": "string", "description": "Ed25519 signature base64 encoded" },
    "payload": { "type": "object", "description": "Canonical snapshot of event data" },
    "timestamp": { "type": "string", "format": "date-time" },
    "veracity_score": { "type": "number", "minimum": 0, "maximum": 1 }
  },
  "required": ["id", "hash", "signer", "signature", "timestamp"]
}


⸻

A.3 Ledger Entry Schema (ledger_entry.schema.json)

{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "LedgerEntry",
  "type": "object",
  "properties": {
    "index": { "type": "integer" },
    "previous_hash": { "type": "string", "pattern": "^[A-Fa-f0-9]{64}$" },
    "current_hash": { "type": "string", "pattern": "^[A-Fa-f0-9]{64}$" },
    "merkle_root": { "type": "string", "pattern": "^[A-Fa-f0-9]{64}$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "quorum_signatures": {
      "type": "array",
      "items": { "type": "string" }
    },
    "receipts": {
      "type": "array",
      "items": { "$ref": "receipt.schema.json" }
    }
  },
  "required": ["index", "previous_hash", "current_hash", "timestamp"]
}


⸻

B. API Contract (FastAPI / REST)

Method	Endpoint	Description	Request Body	Response
POST	/api/scars	Submit new contradiction record	Scar JSON	{ "receipt_id": "uuid", "status": "submitted" }
PATCH	/api/scars/{id}/resolve	Update or resolve a contradiction	{ "status": "resolved", "notes": "..." }	{ "receipt_id": "uuid", "status": "resolved" }
GET	/api/ledger/verify	Verify Merkle chain integrity	None	{ "status": "valid", "root": "hash" }
GET	/api/receipts/{id}	Retrieve a signed receipt	None	Receipt JSON
POST	/api/governance/vote	Submit a quorum vote	{ "ledger_id": "uuid", "signature": "..." }	{ "accepted": true }


⸻

C. Class Interface Stubs (Pseudocode)

class Scar:
    def __init__(self, id, type, description, severity, timestamp, **kwargs):
        ...

class Receipt:
    def __init__(self, id, payload, signer):
        self.hash = self.compute_hash(payload)
        self.signature = self.sign(signer)
        ...

class Ledger:
    def append(self, receipt: Receipt):
        self.entries.append({
            "index": len(self.entries),
            "previous_hash": self.entries[-1]["current_hash"],
            "current_hash": receipt.hash,
            "timestamp": datetime.utcnow(),
            "quorum_signatures": []
        })

class GovernanceKernel:
    def verify_quorum(self, ledger_entry):
        """Validate weighted signatures against quorum threshold."""
        ...


⸻

D. Performance and Resilience Targets

Metric	Target	Description
Ledger Append Latency	≤ 250 ms	Time to hash, sign, and append receipt
Merkle Verification	≤ 100 ms	Proof validation for single entry
Quorum Consensus Time	≤ 1.0 s	Weighted signature resolution
Uptime (federated nodes)	≥ 99.5%	Federation node redundancy guarantee
Audit Throughput	≥ 10k receipts/minute	Sustained verification capacity
Fault Tolerance	Automatic rollback on ledger corruption	Self-healing via last valid Merkle root


⸻

E. Example API Flow

# Submit new contradiction
curl -X POST http://localhost:8080/api/scars \
     -H "Content-Type: application/json" \
     -d '{
           "type": "Logical",
           "description": "Policy claims equality but allocates resources unequally",
           "severity": 9
         }'

# Verify ledger
curl http://localhost:8080/api/ledger/verify
