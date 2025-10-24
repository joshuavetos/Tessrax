# Tessrax — AI Governance & Contradiction Metabolism Framework

[![CI](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml/badge.svg)](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-success.svg)
![Last Commit](https://img.shields.io/github/last-commit/joshuavetos/Tessrax.svg)

> **Keywords:** AI Governance · Contradiction Metabolism · Audit Ledger · Epistemic Integrity · Transparency · Trust Framework

---

## Overview

**Tessrax** transforms contradictions into verified understanding.  
It detects logical and semantic tensions in data or reasoning, metabolizes them through governance protocols, and records each resolution as a cryptographically signed receipt.  

This creates a self-auditing governance architecture—one that learns by reconciling conflict rather than concealing it.

> Every resolution becomes a verifiable artifact. Every artifact strengthens system integrity.

---

## Core Concepts

- **Claim Extractor** – Normalizes raw inputs into structured claims with provenance metadata.
- **Contradiction Engine** – Flags tensions between claims and quantifies their severity.
- **Governance Kernel** – Applies Tessrax protocols to decide on remediation paths.
- **Ledger** – Anchors every governance action to a verifiable, hash-linked receipt.
- **Reconciliation Engine** – Synthesizes clarity statements from detected contradictions and appends them to the audit ledger.

---

## Quick Start

The canonical runtime targets **Python 3.10.x**.

```bash
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax
python3.10 -m venv .venv --prompt tessrax
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python --version            # should report Python 3.10.x
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=$(pwd):$PYTHONPATH
pytest -q

python examples/minimal_detect.py

Full setup instructions are available in docs/getting_started.md.

⸻

Sample Ledger Receipt

{
  "event_type": "CONTRADICTION_RESOLUTION",
  "timestamp": "2025-10-17T13:10:00Z",
  "claim_id": "claim-42",
  "severity": "high",
  "action": "REMEDIATE",
  "clarity_fuel": 7.8,
  "prev_hash": "bf5a3e4cd9e9c82e",
  "hash": "fae2d0485f1cba11",
  "signature": "ed25519:4aa3..."
}

Each record is produced by tessrax.ledger.Ledger after a governance cycle and can be independently verified against the Merkle root.

⸻

System Architecture

Component	Role
Claim Extractor	Normalizes raw text or data into structured claims with provenance metadata.
Contradiction Engine	Detects and quantifies logical, semantic, or normative tension between claims.
Governance Kernel	Applies the four Tessrax Protocols—Memory, Metabolism, Governance, and Trust—to resolve conflicts.
Ledger	Stores tamper-evident receipts, hash-linked for verifiable lineage.
Visualizer	Renders dashboards and audit views for real-time epistemic health monitoring.

Refer to docs/architecture_overview.md for diagrams and flow narratives.

⸻

Repository Structure

Path	Purpose
analysis/	Exploratory research notebooks and prototypes.
docs/	Core documentation (architecture, setup, protocols, developer guide).
examples/	Runnable demonstrations of contradiction detection and governance metabolism.
tessrax/	Installable Python package implementing core engines and ledger.
tests/	Pytest suite covering engines, governance kernel, and ledger verification.
.github/	Automation workflows, issue templates, and contribution policies.


⸻

Examples

Script	Description
examples/minimal_detect.py	Loads sample claims and flags contradictions.
examples/governance_demo.py	Simulates a full governance cycle, producing signed receipts.
examples/dashboard_snapshot.py	Generates a dashboard-ready summary of ledger state.

Run any example from the repository root:

python examples/<script>.py


⸻

## Runtime Environment & Verification

- Supported interpreter: **Python 3.10.x**. GitHub Actions pins this version via the CI matrix and local work should mirror it.
- Dependencies: install with `pip install -r requirements.txt` and refresh `requirements-lock.txt` using `pip freeze > requirements-lock.txt` when dependencies change.
- Import discipline: only absolute `tessrax.*` imports are allowed; `.flake8` enforces this rule.
- Stub data lives in `ledger/ledger.jsonl` and `tessrax/docs/CorporateFrienthropy/company_frienthropy.csv`—replace them before production use.
- Verification commands: export `PYTHONPATH=$(pwd):$PYTHONPATH` and run `pytest -q` to confirm runtime integrity.
- CI mirrors these steps and includes a lock-file diff to keep local and remote environments aligned.


⸻

Testing

Run all tests locally:

pytest

Continuous integration executes the same workflow automatically on each commit
via GitHub Actions (.github/workflows/tests.yml).

⸻

Documentation
   •   Architecture Overview
   •   Getting Started
   •   Governance Protocols
   •   Developer Guide
   •   [Metabolic Reconciliation](docs/metabolic_reconciliation.md)

⸻

Community & Security
   •   Review the Code of Conduct and Contributing Guide before submitting pull requests.
   •   Responsible disclosure via SECURITY.md.
   •   Discussions, research proposals, and governance experiments are welcome through GitHub issues.

⸻

Citation

If referencing Tessrax in research or implementation:

Vetos, J. (2025). Tessrax: A Framework for AI Governance and Contradiction Metabolism.
GitHub. https://github.com/joshuavetos/Tessrax

⸻

Tessrax — turning contradiction into cognition, and cognition into proof.
