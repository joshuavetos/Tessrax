# Tessrax v13 — Formally Verified Contradiction Metabolism System  
*A self-auditing governance organism that converts contradiction into verified knowledge.*

---

## 1. Overview

Tessrax is a living governance framework that learns by resolving its own contradictions.

Instead of ignoring inconsistencies, it treats them as *fuel*. Every conflicting signal—between policies, datasets, or agents—is recorded, analyzed, resolved by quorum, and sealed into a cryptographically verifiable ledger.  
Each contradiction becomes a “Scar,” each resolution a “Receipt,” and together they form an auditable chain of reasoning the system can replay to prove what it learned and why.

Tessrax is built for environments where **trust must be earned, not assumed**: AI governance, institutional compliance, autonomous systems, and any domain where integrity and adaptability coexist.

---

## 2. How Tessrax Thinks

### 2.1 Contradiction Metabolism  
Most systems delete contradictions as bugs. Tessrax metabolizes them.  
When two truths collide, it identifies the tension, scores it for severity and uncertainty, and routes it to the **Metabolism Engine**.  
Contradictions aren’t chaos—they’re energy sources for learning. The stronger the tension, the greater the potential insight.

### 2.2 Epistemic Reset  
Sometimes the system realizes the current worldview is beyond repair.  
When its epistemic instability score crosses a defined threshold (default > 0.75), the **Governance Kernel** triggers an *Epistemic Reset*: a “cold agent” reboot that starts fresh from first principles.  
Nothing is carried over except receipts—the record of what failed.  
It’s the computational equivalent of scientific humility.

### 2.3 The Judgment Learning Loop  
Tessrax doesn’t just follow rules—it learns better judgment.  
It compares its predicted task quality against actual outcomes, computes the error, and adjusts internal weights for metrics like completeness, falsifiability, and contradiction productivity.  
This loop refines when it should *reset*, *refine*, or *continue*, producing what amounts to machine wisdom.

---

## 3. How Tessrax Proves Trust

### 3.1 Immutable Ledger  
Every event—detection, decision, reset—is recorded in an append-only ledger signed with Ed25519 keys.  
Each entry includes:
- SHA-256 hash of prior entry (`previous_hash`)
- Merkle root of all receipts
- Timestamp + signer quorum signatures

This chain guarantees *tamper evidence and full provenance*; no history can be rewritten without detection.

### 3.2 Formal Verification Layer  
Trust isn’t rhetorical—it’s mathematical.  
Tessrax v13 introduces a **Formal Verification Layer** using TLA⁺, Coq, and Alloy to prove that seven invariants always hold:

| ID | Invariant | Guarantee |
|----|------------|-----------|
| L1 | Hash-Chain Integrity | No broken lineage between ledger entries |
| L2 | Merkle Root Consistency | Receipts tree matches recorded root |
| L3 | Signature Validity | Every receipt verifiably signed |
| G1 | Weighted Quorum | Governance actions meet charter threshold |
| G2 | Revocation Propagation | Key revocations propagate < 1 s |
| G3 | Contradiction Closure | Every resolved Scar has an audit receipt |
| T2 | Fork Resistance | No duplicate ledger indices with differing hashes |

If all invariants hold, Tessrax cannot drift, fork, or silently corrupt itself.

---

## 4. Architecture Overview

| Layer | Components | Function |
|-------|-------------|----------|
| **Memory Engine** | `memory.py` | Reconstructs current state from ledger receipts |
| **Metabolism Engine** | `contradiction_engine.py`, `semantic_analyzer.py`, `metabolism_adapter.py` | Detects and processes contradictions |
| **Governance Kernel** | `governance_kernel.py`, `ledger.py`, `receipts.py` | Quorum logic, signatures, policy enforcement |
| **Trust Engine** | `world_receipt_protocol.py`, `formal_verify.py` | Cryptographic proofs, verification API |
| **Visualization Stack** | `dashboard/app.py`, `visualization.py` | Real-time contradiction and ledger graph |
| **Runtime Interface** | `current.py` | Unified launch (FastAPI 8080 + Streamlit 8501) |

**Run:**  
```bash
python current.py

→ FastAPI on 8080 for API, Streamlit dashboard on 8501 for live visualization.

⸻

5. Quick Start

# 1. Build and run the full system
docker compose up -d --build

# 2. Generate contradictions and receipts
python core/contradiction_engine.py

# 3. Verify ledger integrity
python verify_all.py

Expected result:

✅ All invariants hold — Ledger drift: 0%, Forks: 0%, Quorum valid.


⸻

6. Core Schemas & API

Minimal examples (full JSON in /docs/specs/):

Object	Key Fields	Purpose
Scar	id, type, description, severity, timestamp	Encodes a detected contradiction
Receipt	id, hash, signature, payload	Signed proof of event
LedgerEntry	index, previous_hash, merkle_root, quorum_signatures	Immutable audit record

REST endpoints

Method	Endpoint	Description
POST	/api/scars	Submit new contradiction
PATCH	/api/scars/{id}/resolve	Resolve or update
GET	/api/ledger/verify	Validate full Merkle chain
GET	/api/receipts/{id}	Retrieve signed receipt
POST	/api/governance/vote	Cast quorum vote


⸻

7. Cold-Agent Prompting Protocol

Tessrax’s cognitive discipline for governed reasoning:
	1.	Epistemic Reset — start from zero; no hidden bias.
	2.	Declarative Build Contracts — every prompt defines full deliverables.
	3.	Contradiction as Fuel — conflicts feed the next iteration.
	4.	Governance Frame Embedding — every generation occurs under signature and ledger context.
	5.	Receipts-First Discipline — outputs are complete, auditable artifacts.
	6.	Governed Autonomy Bandwidth — reasoning freedom is policy-bound, not random.
	7.	Auditability as Primitive — traceability prioritized over fluency.

Result: every response is a verifiable artifact, every contradiction a metabolic event.

⸻

8. Security & Reliability Targets

Metric	Target	Description
API Authentication Failures	≤ 0.01 %	threshold → audit event
Key Revocation Propagation	≤ 1 s	federation sync
Ledger Consistency Drift	0 %	enforced by hash chain
Node Recovery Time	≤ 30 s	replay from receipts
Test Coverage	≥ 85 %	merge-approval threshold

Security objective: make all trust falsifiable. Tessrax assumes compromise and designs for verifiable recovery, not blind prevention.

⸻

9. Research Extensions

Project	Purpose
Moral Systems Engineering (MSE)	Quantifies moral health dynamics via entropy differentials
Corporate Frienthropy	Maps ethical capital vs. profit contradictions
Contrastive Self-Verification (CSV)	Forces model to generate falsifiable counterfactuals
Domain Packs	AI Memory, Attention Economy, Climate Policy, Democratic Governance


⸻

10. Verification Workflow

Continuous Proof Loop

# Temporal logic check
tla -config formal/tessrax_ledger.cfg formal/tessrax_ledger.tla

# Logical proof
coqc formal/tessrax_ledger.v

# Relational analysis
java -jar alloy.jar formal/tessrax_ledger.als

Outputs are hashed into /ledger/formal_receipts.jsonl
→ included in the Merkle chain for full audit trace.

⸻

11. Repository Map

tessrax/
├── core/              # Engines: memory, metabolism, governance, trust
├── dashboard/         # Streamlit + D3 visualization
├── formal/            # TLA⁺, Coq, Alloy proofs
├── docs/specs/        # Schemas, API, invariants
├── tests/             # pytest + invariant verification
├── compliance/        # charters, policy rules
└── current.py         # unified runtime launcher


⸻

12. License & Attribution

License: MIT – Open Research Variant (2025)
Author: Joshua Scott Vetos
Entity: Tessrax LLC
Version: v13.0
Repository: github.com/joshuavetos/Tessrax
Integrity Anchor: GPT to Josh — Tessrax LLC

⸻

“If existence is contradiction, then consistency is proof of life.”
— Tessrax LLC (2025)
