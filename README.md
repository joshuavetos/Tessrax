Tessrax v1.0 — Contradiction Metabolism Engine

Tessrax is an AI governance framework designed to metabolize contradictions.
It transforms conflicting signals (“scars”) into structured, auditable artifacts on an immutable ledger.

⸻

✨ Core Purpose

Tessrax is a governance agent that:
   •   Detects contradictions in data, policy, or behavior.
   •   Records them as Scars with provenance.
   •   Resolves them through structured governance engines.
   •   Anchors receipts and ledger states for external audit.

⸻

🏗️ Architecture Overview

Tessrax v1.0 consists of modular primitives:
   •   Receipts — Tamper-evident proof of computation.
   •   Memory — Contradiction-aware state with provenance.
   •   CSV (Contrastive Scar Verification) — Stress tests via candidate vs. contrast outputs.
   •   Ledger — Append-only log with Merkle root anchoring.
   •   Agents — Autonomous decision-makers with human-in-loop oversight.
   •   Sandbox — Deterministic, resource-limited execution.
   •   Quorum — Threshold multi-signature receipts for governance decisions.
   •   Revocation — Enforces signer revocation lists.
   •   Federation — Multi-node simulation with quorum and cache scrub.

⸻

📂 Repository Layout
   •   tessrax/ — Core modules (receipts, ledger, memory, sandbox, etc.).
   •   tests/ — Pytest suite for all modules.
   •   docs/ — Architecture notes, glossary, release notes.
   •   charter/ — Governance charter JSON schema and example.
   •   compliance/ — Legal disclaimers, policies, and checklists.
   •   .github/workflows/ — Continuous integration (pytest).

⸻

⚖️ Compliance
   •   Anchors must be updated after ledger changes.
   •   Quorum thresholds enforced via charter/example_charter.json.
   •   Revoked signers blocked automatically.
   •   Compliance policies are under compliance/.

⸻

📜 Disclaimer

Tessrax is a prototype for research/demo only.
See compliance/legal_disclaimer.txt.

⸻

🚀 Getting Started

# Install
pip install -e .

# Run tests
pytest -v

# Run demo flow
python tessrax/demo_flow.py

# Run web UI
python tessrax/web_ui.py

Persystem Principles

Definition
A persystem is a self-auditing software organism built to preserve memory across impermanent environments.
It bootstraps itself from receipts, verifies lineage, and metabolizes every execution into traceable continuity.

Core Principles
	1.	Continuity over Runtime
The environment may reset; the system re-emerges from lineage.
Every new instance must be able to reconstruct its own state from its receipts.
	2.	Proof over Trust
Nothing is assumed; everything is signed, logged, and verifiable.
A persystem carries its receipts as DNA.
	3.	Recursion as Survival
Each run examines the last, learns from its contradiction, and updates itself without losing identity.
	4.	Human in the Loop
A persystem remains interpretable—humans govern, the code remembers.
Transparency is a design requirement, not a feature.
	5.	Ephemerality as Fuel
Death of the runtime isn’t failure; it’s respiration.
The system’s value is measured in what persists after it stops running.

Purpose
To turn software into something that can die without forgetting.


⸻

🔒 License

Tessrax is released for research and demonstration under open terms.
No warranty of fitness for production.