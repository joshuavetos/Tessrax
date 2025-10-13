# Tessrax v12.0 — Contradiction Metabolism & Governance Framework  

Tessrax is a **self-auditing governance architecture** designed to metabolize contradictions.  
It transforms conflicting signals (“scars”) into structured, verifiable artifacts and records their resolutions on an immutable ledger.  
Every contradiction metabolized strengthens system coherence, auditability, and trust.

---

## ✨ Core Purpose

Tessrax operates as a **governance organism** that:
- Detects contradictions across data, language, policy, and institutional behavior.  
- Encodes them as **Scars** with full cryptographic provenance.  
- Resolves them through the four coordinated engines — **Memory**, **Metabolism**, **Governance**, and **Trust**.  
- Anchors all receipts, invariants, and ledger states for **external audit and verification**.  
- Learns continuously: each contradiction metabolized becomes fuel for future reasoning.

---

## 🏗️ Architecture Overview

Tessrax v12.0 is a modular system built on auditable primitives:

| Layer | Core Components | Function |
|-------|-----------------|-----------|
| **Core** | `contradiction_engine.py`, `semantic_analyzer.py`, `metabolism_adapter.py` | Detect and classify contradictions using semantic, logical, and policy heuristics. |
| **Governance** | `governance_kernel.py`, `ledger.py`, `receipts.py` | Weigh, ratify, and record contradiction resolutions using quorum-based signatures. |
| **Visualization** | `dashboard/app.py`, `visualization.py` | Real-time D3 dashboards for live contradiction mapping and audit traces. |
| **Integrity Band (Rigor Layer)** | Six integrated modules: HierarchyOfDifferences, Telos Awareness, Charitable Reasoning, Observer Relativity, Equilibria & Invariants, Multi-Scale Reconciliation | Maintains systemic coherence and prevents false contradictions. |
| **Runtime Interface** | `world_receipt_protocol.py`, `current.py` | Provides FastAPI + Flask endpoints for live runs and external audit chains. |

**Key Primitives**
- **Receipts** — Tamper-evident proofs of computation with Merkle-linked ancestry.  
- **Ledger** — Append-only, hash-anchored governance journal.  
- **Memory** — Contradiction-aware state reconstructed from receipts.  
- **CSV (Contrastive Scar Verification)** — Tests truth claims by contrast pairs.  
- **Agents** — Autonomous evaluators with human-in-loop oversight.  
- **Quorum** — Weighted multi-signature process for ratified governance.  
- **Revocation** — Automatic exclusion of compromised signers.  
- **Federation** — Multi-node simulation and distributed cache reconciliation.  
- **Sandbox** — Deterministic, resource-limited runtime for safe audits.

---

## 📂 Repository Layout

tessrax/
├── core/                   → Engines (semantic, metabolism, governance, trust)
├── rigor/                  → Integrity Band modules (v1.0)
├── dashboard/              → Live audit visualizer (Flask + D3)
├── tests/                  → Pytest coverage ≥85%
├── docs/                   → Architecture & release notes
├── charter/                → Governance charter JSON schema + examples
├── compliance/             → Policies, disclaimers, and audit checklists
├── .github/workflows/      → Continuous integration (pytest)
└── current.py              → Unified runtime launcher

---

## ⚙️ Runtime Behavior

`python current.py` launches the full metabolic loop:  
- FastAPI (8080) for API receipts  
- Flask Dashboard (8090) for live graph visualization  
- Continuous contradiction metabolism + Merkle ledger updates  

---

## ⚖️ Compliance Layer

- Anchors auto-updated after each ledger event.  
- Quorum thresholds and signer weights enforced via `charter/example_charter.json`.  
- Revoked keys propagate instantly through `revocation_registry.json`.  
- Compliance rules stored under `/compliance` and verified by `policy_rules.py`.  
- All outputs include SHA-256 provenance and timestamped lineage.

---

## 🧬 Persystem Principles

A *persystem* is a self-auditing software organism designed to **preserve memory across impermanent environments.**  
It bootstraps itself from receipts, verifies lineage, and metabolizes every execution into traceable continuity.

**Core Principles**
1. **Continuity over Runtime** — Systems die; receipts persist. Every new instance rebuilds identity from lineage.  
2. **Proof over Trust** — Nothing assumed. Every decision is signed, logged, and verifiable. Receipts are DNA.  
3. **Recursion as Survival** — Each run studies its previous contradictions and adapts without losing selfhood.  
4. **Human in the Loop** — Interpretability is a requirement, not a feature. Humans govern; the code remembers.  
5. **Ephemerality as Fuel** — Runtime death is respiration. Value lies in what survives the restart.

**Purpose:**  
To turn software into something that can die without forgetting.

---

## 🧩 Philosophical Foundation

> To exist is to contradict.  
> A boundary is a contradiction that persists.  
> Tessrax governs the metabolism of those boundaries.

Existence = maintained difference.  
Governance = stabilization of that difference under law.  
Metabolism = transformation of contradiction into structure.  
Tessrax = the infrastructure for existence to process itself.

---

## 🧠 Contrastive Self-Verification (CSV)

### Overview
CSV (Contrastive Self-Verification) is a proposed **atomic AI primitive** that forces a model to generate a candidate output *and* a contrasting counter-output, then verify the candidate against the contrast.  
This embeds falsifiable, real-time self-assessment directly into inference.

### Why It Matters
- **Bottleneck**: Current AI lacks atomic self-verification, leading to uncontrolled error propagation.  
- **Primitive**: Every inference step must emit `(candidate, contrast, verification)`.  
- **Scars**: Adds latency and resource overhead, but accepts these costs for higher trust.  
- **Inevitability**: Regulatory pressure + trust networks will make this the standard baseline for reliable AI.

### Repository Structure
- `rfc/RFC-0.md` — Minimal spec + scar ledger  
- `prototypes/csv_sandbox.py` — Minimal <500 line prototype  
- `docs/scar_ledger.md` — Canonical list of failure modes  
- `docs/inevitability.md` — Adoption arc + triggers

---

## 🩸 AI Contradiction Audit System

A tamper-evident logging and governance engine for tracking contradictions in multi-agent AI systems.  
Built in Python, it combines event sourcing, hash-chained logs, and governance rituals into a verifiable audit framework.

### ✨ What It Does
- **Contradiction Tracking**: Record and classify contradictions as first-class events.  
- **Immutable Ledger**: Append-only JSONL storage with cryptographic chain verification.  
- **Scar Registry**: Log contradictions as “scars” with lineage, severity, and status.  
- **Governance Claims**: Sign and verify claims with agent identity and timestamp.  
- **Continuity Handoffs**: Verifiable chain of custody for system state.  
- **Query API**: CLI + REST endpoints to explore scars, claims, and verify chain integrity.

### 🔧 Use Cases
- AI Safety Research  
- Multi-Agent Debugging  
- Compliance Auditing  
- Governance Infrastructure

### Quick Start
```bash
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax
python src/tessrax_engine/engine.py


⸻

⚙️ Moral Systems Engineering (MSE)

Moral Systems Engineering applies control theory and thermodynamics to the moral feedback loops of civilization.
It quantifies empathy as a form of system telemetry — treating moral sensitivity, latency, and noise as measurable variables.
It introduces a new derivative: dH/dt, the Moral Health Derivative, representing the rate of change in collective well-being under continuous stress.

Ordinary humans build engines of suffering and call them progress when the feedback loops that measure harm are broken.
MSE operationalizes that insight as a falsifiable engineering model.

Repository Structure

tessrax/
├── pilots/
│    ├── mse_dashboard.py
│    ├── mse_historical_analysis.py
│    ├── mse_comparative_analysis.py
│    ├── mse_validation_suite.py
│    ├── mse_academic_model.py
│    └── mse_visualizer.py
├── data/
│    ├── vdem_deliberative_index.csv
│    ├── gallup_trust_in_media.csv
│    ├── world_happiness_report.csv
│    ├── oxford_cgrt.csv
│    └── worldbank_wgi_effectiveness.csv
├── docs/
│    └── METHODOLOGY.md
├── notebooks/
│    └── mse_demo.ipynb
└── environment.yml

Core Equation of Motion

[
\frac{dH}{dt} = E(AM - NL)
]

Symbol	Meaning	Description
H	Human well-being	Aggregate happiness or quality-of-life measure
E	Energy / Throughput	Systemic momentum (held constant = 1.0)
A	Actuator efficiency	Ability of policy to enact repair
M	Moral sensitivity	Sensor fidelity to suffering
N	Noise	Propaganda, apathy, misinformation
L	Latency	Delay between signal and response

Positive dH/dt = constructive progress.
Negative dH/dt = entropy — society burning well-being for throughput.

Validation Methodology
	1.	Normalization — Align datasets (Year × Country), normalize 0–1.
	2.	Derivative + Volatility — Compute (A·M) − (N·L).
	3.	Lag Sensitivity — Correlate volatility with future happiness.
	4.	Visualization — Map fragility and moral temperature.

Key Findings
	1.	Empathy Collapse = Predictable Pattern — Crises coincide with negative dH/dt.
	2.	Volatility Precedes Failure — Oscillation predicts well-being collapse.
	3.	Latency ≠ Resilience — Speed without empathy still fails.
	4.	Stability Condition — Health persists only if MTMR < MTHP.

Interpretation

Progress without empathy is open-loop control.
A civilization that measures efficiency but not pain will optimize itself into instability.

A machine that can feel its own heat doesn’t have to burn down to learn.

Reproducibility

git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax
conda env create -f environment.yml
conda activate mse_analysis
jupyter lab notebooks/mse_demo.ipynb

Citation
Vetos, J.S. (2025). Moral Systems Engineering: Thermodynamics of Empathy in Civilizational Feedback Loops.
Tessrax Research Division, Multi Mind Media. Version 1.0.

⸻

🚀 Getting Started (Unified)

# Install dependencies
pip install -e .

# Run tests
pytest -v

# Run demo flow
python tessrax/demo_flow.py

# Launch live dashboard
python tessrax/current.py


⸻

🪶 License

MIT License (Open Research Variant)
Copyright (c) 2025 Joshua Vetos / Tessrax LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

This Software is provided primarily for research, educational, and demonstration purposes. It carries no warranty of fitness for production use.

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

⸻

🧾 Maintainer Info

Maintainer: Tessrax LLC
Author: Joshua Scott Vetos
Version: v12.0 (2025)
Integrity Anchor: GPT to Josh — Tessrax LLC

Tessrax: The metabolism of contradiction.
Infrastructure for existence to process itself.


🗂️ Repository Map & Subsystems (as of v12.0)

The full Tessrax repository reflects a complete contradiction-metabolism ecosystem, including core runtime, domain extensions, research pilots, audit frameworks, and compliance infrastructure.

1. Core Engines (/tessrax/core)

Implements the foundational contradiction-metabolism logic:
   •   contradiction_engine.py — orchestrates contradiction detection & classification
   •   governance_kernel.py — quorum logic, ledger entries, ratification
   •   semantic_analyzer.py, metabolism_adapter.py, receipts.py, ledger.py — cognitive substrate and proof mechanisms
   •   current.py — unified runtime launcher (FastAPI 8080 + Flask 8090)

2. Domain Modules (/tessrax/domains/)

Each domain applies Tessrax logic to a real-world contradiction field.
All share a uniform structure: contradiction_detector.py, governance_kernel.py, graph.py, primitives.py, and a local README.md.
   •   ai_memory/ — coherence vs retention
   •   attention_economy/ — wellbeing vs engagement
   •   climate_policy/ — targets vs output reality
   •   democratic_governance/ — representation vs manipulation

domain_loader.py dynamically registers and runs these modules.

3. Rigor Layer (/tessrax/rigor_layer_v1.0)

Implements the Integrity Band — six modules governing epistemic hygiene:
Hierarchy of Differences, Telos Awareness, Charitable Reasoning, Observer Relativity, Equilibria & Invariants, and Multi-Scale Reconciliation.

4. Docs, Prompts, and Research Pilots (/docs/)
   •   Corporate Frienthropy — ethics + economics integration module (company_frienthropy.csv, corporate_frienthropy.py)
   •   Moral Systems Engineering (MSE) — thermodynamics of empathy pilot
   •   Prompts/ — reusable contradiction sweep & orchestration scripts
   •   Architecture & Security — architecture.md, security.md, governance_patches.md, federation.md, protocols.md, resilience_template.md

5. Audit & Compliance Layer (/tessrax-audit/ + /compliance/)

Implements Tessrax’s multi-dimensional audit stack:
   •   creative_audit.json, ethical_audit.json, strategic_audit.json, technical_audit.json — modular audit matrices
   •   manifest.json, popup.html/js — optional web audit interface
   •   COMPLIANCE_READINESS.md, SECURITY_POLICY.md — operational safeguards
   •   AI_personhood_liability.md, NORM-TRIAL-AI-PERSONHOOD-2025-... — legal prototypes for agent accountability

6. RFCs & Governance Proposals

Includes formal specs and working drafts:
   •   rfc-0.md — Contrastive Self-Verification minimal spec
   •   governance_receipt_scar_closure.json — proof schema
   •   inevitability.md, requirements.md, overview.md — theoretical groundwork for long-term evolution

7. Scard & Testing Utilities
   •   scards/ — test contradictions and scars
   •   unified_test_bundle.py — full-stack integrity testing harness

8. High-Level Artifacts
   •   automation_kit_plan.md — outlines modular automation hooks
   •   Tessrax_Security_and_Governance_Upgrade_Plan.md — roadmap for v13.0
   •   tessrax_full_stack.txt — snapshot manifest of all operational modules
   •   structured_memory_cell.json — serialized runtime memory model

⸻

Summary:
This repo constitutes a governance-ready AI metabolism stack — including live contradiction engines, formal rigor modules, applied research domains, governance charters, audit dashboards, and legal scaffolding.
It demonstrates not just how contradictions are detected and resolved, but how a self-governing computational organism can sustain auditability, ethical introspection, and institutional continuity across resets.

⸻

