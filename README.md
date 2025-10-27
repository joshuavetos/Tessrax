# Tessrax — AI Governance Through Contradiction Metabolism

[![CI](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml/badge.svg)](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-success.svg)
![Last Commit](https://img.shields.io/github/last-commit/joshuavetos/Tessrax.svg)

> **Keywords:** AI Governance · Epistemic Integrity · Contradiction Detection · Audit Ledger · Runtime Trust · Verifiable Reasoning

---

## 🧠 What Is Tessrax?

**Tessrax is a governance engine that turns contradiction into cognition.**

It detects logical and semantic conflicts across data or claims, metabolizes them through formal governance protocols, and logs every resolution as a signed, hash-linked ledger receipt.

> ⚖️ Contradiction is not a bug — it’s a signal. Tessrax governs it, audits it, and turns it into verified clarity.

Use Tessrax to:
- Detect and reconcile contradictions in claims, metrics, or agent outputs
- Enforce epistemic integrity across AI or multi-agent systems
- Generate a cryptographically signed audit trail of all governance decisions

---

## 🔧 Core System Components

| Component             | Role |
|-----------------------|------|
| **Claim Extractor**   | Normalizes raw inputs into structured, provenance-rich claims |
| **Contradiction Engine** | Flags and quantifies tensions between claims |
| **Governance Kernel** | Applies Tessrax protocols to determine remediation paths |
| **Reconciliation Engine** | Synthesizes clarity statements and appends audit records |
| **Ledger**            | Verifiable, hash-linked, tamper-evident receipt log |
| **Dashboard**         | Real-time view of epistemic metrics and contradiction metabolism |

See [`docs/architecture_overview.md`](docs/architecture_overview.md) for detailed diagrams.

---

## 🚀 Quick Start

Tessrax requires **Python 3.10.x**.

```bash
# Clone the repo
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax

# Set up environment
python3.10 -m venv .venv --prompt tessrax
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run tests and a sample contradiction detection
pytest -q
python examples/minimal_detect.py

📄 Full setup: docs/getting_started.md

⸻

🧾 Sample Governance Receipt

After each governance cycle, Tessrax appends a signed receipt to the audit ledger:

{
  "event_type": "CONTRADICTION_RESOLUTION",
  "timestamp": "2025-10-17T13:10:00Z",
  "claim_id": "claim-42",
  "severity": "high",
  "action": "REMEDIATE",
  "clarity_fuel": 7.8,
  "hash": "fae2d0485f1cba11",
  "signature": "ed25519:4aa3..."
}

All receipts are independently verifiable and anchored in a Merkle-linked ledger chain.

⸻

🧪 Demonstrations

Script	Description
examples/minimal_detect.py	Loads test claims and flags contradictions
examples/governance_demo.py	Simulates a full governance cycle with ledger outputs
examples/dashboard_snapshot.py	Generates a Streamlit-compatible metrics snapshot

Run any demo from the project root:

python examples/<script>.py


⸻

🧬 Runtime & Integrity
   •   ✅ Python version: 3.10.x (enforced by CI and local lockfiles)
   •   🔒 Dependency lock: requirements-lock.txt, auto-generated via automation/regenerate_lock.sh
   •   📦 Absolute imports enforced (tessrax.*) via .flake8
   •   🔍 Tests: pytest -q, same as CI runner
   •   🧪 Self-test: python -m tessrax.metabolism.reconcile --self-test
   •   📁 Stub data: ledger/ledger.jsonl, company_frienthropy.csv (replace for production)

⸻

🧩 System Architecture

Concept	Description
Memory	Canonical record of all claims and contradictions
Metabolism	Conversion of contradictions into clarity statements
Governance	Decision logic over epistemic uncertainty
Trust	Signed receipts, ledger verification, provenance guarantees

Refer to docs/architecture_overview.md and docs/metabolic_reconciliation.md

⸻

📁 Repository Structure

Path	Purpose
tessrax/	Core engines, reconciliation kernel, ledger logic
examples/	Demonstrations and CLI simulations
docs/	Architecture, governance protocols, audit flows
tests/	Pytest suite for all subsystems
automation/	Lockfile guards, meta-launcher logic, CI tools
dashboard/	Streamlit dashboard templates and visualizers


⸻

🧱 Verification & CI

Tessrax runs a complete reproducibility audit on every commit:
   •   ✅ GitHub Actions (.github/workflows/tests.yml)
   •   ✅ Lockfile consistency guard (sync_guard.py)
   •   ✅ Epistemic metrics validator (integrity, drift, severity)
   •   ✅ Adversarial contradiction testing and resilience simulations

⸻

🛡️ Community, Security & Contributions
   •   📜 Code of Conduct
   •   🧪 Contributing Guide
   •   🔐 Security Policy
   •   🧠 Open to issues, proposals, and experimental forks

⸻

🔬 Citation

If referencing in research or implementation:

Vetos, J. (2025). Tessrax: A Framework for AI Governance and Contradiction Metabolism.
GitHub. https://github.com/joshuavetos/Tessrax

⸻

🧭 Why Tessrax Exists

Modern AI systems are increasingly non-deterministic, multi-agent, and epistemically unstable.
Contradictions are inevitable. Tessrax turns that inevitability into architecture:
   •   It doesn’t suppress conflict — it resolves and records it.
   •   It doesn’t trust blindly — it signs, verifies, and proves.
   •   It metabolizes contradiction into governance clarity.

Tessrax — turning contradiction into cognition, and cognition into proof.
