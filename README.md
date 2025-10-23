# Tessrax – AI Governance & Contradiction Metabolism Framework

[![CI](https://github.com/YOUR_ORG/Tessrax/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_ORG/Tessrax/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-success.svg)
![Last Commit](https://img.shields.io/github/last-commit/YOUR_ORG/Tessrax.svg)

> **Keywords:** AI Governance · Contradiction Metabolism · Audit Ledger · Transparency · Trust Framework · Tessrax

Tessrax converts systemic contradictions into auditable knowledge. Inputs are transformed into structured claims, scored for tension, metabolised through governance protocols, and anchored in an immutable ledger for transparent review.


## Table of Contents

1. [Quick Start](#quick-start)
2. [Sample Ledger Receipt](#sample-ledger-receipt)
3. [Repository Structure](#repository-structure)
4. [Core Concepts](#core-concepts)
5. [Documentation](#documentation)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Community & Security](#community--security)

## Quick Start

```bash
git clone https://github.com/YOUR_ORG/Tessrax.git
cd Tessrax
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
python examples/minimal_detect.py
```

For full setup guidance, visit [`docs/getting_started.md`](docs/getting_started.md).

## Sample Ledger Receipt

```json
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
```

Receipts are produced by `tessrax.ledger.Ledger` when governance cycles complete.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `analysis/` | Historical research scripts, notebooks, and exploratory experiments. |
| `docs/` | Architecture, onboarding, governance protocols, and developer documentation. |
| `examples/` | Runnable demonstrations of contradiction detection, governance simulation, and dashboard snapshots. |
| `tessrax/` | Installable Python package implementing extractors, engines, governance kernel, and ledger. |
| `tests/` | Pytest-based regression suite covering contradiction detection and ledger verification. |
| `.github/` | Issue templates and automation workflows. |

## Core Concepts

- **Claim Extractor** – Normalises raw input into structured claims with provenance metadata.
- **Contradiction Engine** – Quantifies tension between claims to prioritise governance attention.
- **Governance Kernel** – Applies Memory, Metabolism, Governance, and Trust protocols to resolve contradictions.
- **Ledger** – Appends tamper-evident receipts capturing every governance decision.
- **Visualizer** – Summaries and dashboards communicating status to auditors and stakeholders.

An end-to-end architecture diagram is available in [`docs/architecture_overview.md`](docs/architecture_overview.md).

## Documentation

- [Architecture Overview](docs/architecture_overview.md)
- [Getting Started](docs/getting_started.md)
- [Governance Protocols](docs/governance_protocols.md)
- [Developer Guide](docs/developer_guide.md)

## Examples

| Script | Description |
| --- | --- |
| `examples/minimal_detect.py` | Load sample claims and highlight contradictions. |
| `examples/governance_demo.py` | Run a deterministic governance cycle that records receipts. |
| `examples/dashboard_snapshot.py` | Generate a dashboard-ready snapshot summarising ledger health. |

Run any script with `python examples/<script>.py` from the repository root.

## Testing

Execute the full suite with:

```bash
pytest
```

Continuous integration runs the same workflow on GitHub Actions (`.github/workflows/tests.yml`).

## Community & Security

- Review the [Code of Conduct](CODE_OF_CONDUCT.md) and [Contributing Guide](CONTRIBUTING.md) before opening pull requests.
- Report security vulnerabilities responsibly via [SECURITY.md](SECURITY.md).
- Join discussions through GitHub issues and proposals to help steer Tessrax governance research.
