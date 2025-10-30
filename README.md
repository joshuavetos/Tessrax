# Tessrax — AI Governance Through Contradiction Metabolism

[![CI](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml/badge.svg)](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-brightgreen.svg)
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

> ℹ️  Need a lightweight deployment? See [README_CORE](README_CORE.md) for the audited runtime
and [README_ENTERPRISE](README_ENTERPRISE.md) for optional queueing and billing modules.

---

## 🚀 Quick Start

Tessrax requires **Python 3.11.x**.

```bash
# Clone the repo
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax

# Set up environment
python3.11 -m venv .venv --prompt tessrax
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .[dev]
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run tests and a sample contradiction detection
pytest -q

# Install enterprise extras if required
pip install -e .[enterprise]
 
See [`docs/enterprise_migration.md`](docs/enterprise_migration.md) for guidance on the
v15 → v16 enterprise upgrade path and the temporary Python 3.10 support policy.
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
   •   ✅ Python version: 3.11.x (enforced by CI and local lockfiles)
   •   🔒 Dependency lock: requirements-lock.txt, auto-generated via automation/regenerate_lock.sh
   •   📦 Absolute imports enforced (tessrax.*) via .flake8
   •   🔍 Tests: pytest -q, same as CI runner
   •   🧪 Self-test: python -m tessrax.metabolism.reconcile --self-test
   •   📁 Stub data: ledger/ledger.jsonl, company_frienthropy.csv (replace for production)

⸻

## 🗂️ Field Evidence Archive (2025-10-28)

Tessrax ships with the **Field Evidence Archive (2025-10-28)**, a machine-readable JSONL dataset that captures on-the-ground audits, contradiction analyses, and policy alignment checkpoints gathered during field deployments. The archive powers simulation benchmarks inside the governance kernel and feeds the audit and compliance modules with immutable, provenance-rich evidence packets.

Each record in `tessrax/data/evidence/field_evidence_archive_2025-10-28.jsonl` follows this schema:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Stable identifier for the evidence entry |
| `category` | `str` | Governance domain (e.g., supply_chain, governance, safety) |
| `year` | `int` | Calendar year the evidence was captured |
| `source_type` | `str` | Provenance channel such as field_report, audit log, or simulation |
| `summary` | `str` | Narrative synopsis of the evidence |
| `key_findings` | `list[str]` | Bullet-style findings extracted from the source |
| `alignment` | `dict` | Policy alignment metadata including `policy_reference`, status, and score |
| `citations` | `list[str]` | Human-readable citations or references |

The archive is immutable—subsequent updates should be delivered as new timestamped JSONL files under `tessrax/data/evidence/`. Provenance is maintained through field operations memos, regulatory site visits, and governance simulations recorded by Tessrax teams between 2023 and 2025.

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

## ai_skills Prompt Toolkit

The repository now bundles a hermetic prompt-engineering module named `ai_skills`.
It relies solely on the Python standard library and exposes deterministic prompt
templates, a template renderer, and an evaluation CLI.

### Rendering the Socratic Debugger Template

```bash
python -m ai_skills.prompting.cli render --template socratic_debugger --task "Add numbers" --context "3 and 5"
```

Expected output:

```
"""
ROLE: You are a careful reasoner. Work step-by-step and show checks.
TASK: Add numbers
CONTEXT: 3 and 5
STEPS:
1) Extract claims (bulleted).
2) For each claim, note confidence and what would falsify it.
3) Produce an answer and a short "Why I might be wrong" section.
OUTPUT: Answer, then Checks.
"""
```

### Scoring a Guess Against the Truth Reference

```bash
python -m ai_skills.prompting.cli score --guess "test" --truth "test"
```

### Full Validation Checklist

```bash
pytest tests/ -v
python -m ai_skills.prompting.cli render --template socratic_debugger --task "test" --context "test"
python -m ai_skills.prompting.cli score --guess "test" --truth "test"
```

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
