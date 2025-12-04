# Tessrax — Contradiction Governance Engine

[![CI](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml/badge.svg)](https://github.com/joshuavetos/Tessrax/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-success.svg)
![Last Commit](https://img.shields.io/github/last-commit/joshuavetos/Tessrax.svg)

# Overview

Tessrax detects contradictions in data, models, or agent outputs and records how they are resolved. Reconciliations are logged as signed, hash-linked receipts so decisions can be audited later. The project includes a core governance engine, optional integrations, and example scripts.

Key capabilities:
- Detect logical or semantic conflicts in AI or multi-agent systems.
- Apply policy and quorum rules to reconcile conflicts.
- Produce cryptographically signed ledger entries for auditability.

## Core Architecture

| Layer | Purpose |
|-------|---------|
| Memory Engine | Stores claims, contradictions, and reconciliation logs |
| Metabolism Engine | Converts contradictions into structured clarity statements |
| Governance Kernel | Applies policy and quorum logic to conflicting data |
| Trust Engine | Signs and verifies ledger entries |
| Dashboard | Visualizes contradictions, resolutions, and related metrics |

The architecture diagram is available in `docs/assets/architecture_overview.svg`.

## Quick Start

Tessrax requires **Python 3.11.x**.

```bash
# Clone the repo
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax

# Create and activate a virtual environment
python3.11 -m venv .venv --prompt tessrax
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Verify install
pytest -q

# Optional: enterprise extensions
pip install -e .[enterprise]
```

For full setup and configuration, see [`docs/getting_started.md`](docs/getting_started.md).

## Governance Receipts

Each governance cycle ends with a cryptographically signed receipt recorded in the audit ledger:

```json
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
```

Receipts are Merkle-linked and signed so they can be independently verified.

## Demonstrations

| Script | Description |
|---------|-------------|
| `examples/minimal_detect.py` | Loads sample claims and flags contradictions |
| `examples/governance_demo.py` | Runs a full governance cycle and appends ledger receipts |
| `examples/dashboard_snapshot.py` | Produces a Streamlit-ready visualization snapshot |

```bash
python examples/<script>.py
```

## Runtime Integrity

| Check | Method |
|--------|--------|
| ✅ Python | 3.11.x (enforced by CI) |
| 🔒 Lockfiles | `requirements-lock.txt` auto-generated via `automation/regenerate_lock.sh` |
| 🧠 Imports | Absolute (`tessrax.*`) enforced via `.flake8` |
| 🧪 Tests | `pytest -q` |
| 🧬 Self-Test | `python -m tessrax.metabolism.reconcile --self-test` |

All test results and receipts are mirrored in the audit ledger under `ledger/ledger.jsonl`.

## Field Evidence Archive (2025-10-28)

Tessrax ships with a JSONL dataset of real and simulated contradiction audits gathered during 2023–2025. It is used for simulation benchmarks and compliance metrics inside the governance kernel.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Stable identifier |
| `category` | `str` | Domain (e.g. `supply_chain`, `safety`) |
| `source_type` | `str` | Provenance (`field_report`, `simulation`, etc.) |
| `summary` | `str` | Narrative synopsis |
| `alignment` | `dict` | Policy reference, status, and score |

Located at `tessrax/data/evidence/field_evidence_archive_2025-10-28.jsonl`.

## Verification & Continuous Integration

Tessrax performs a full reproducibility audit on every commit:

- ✅ **GitHub Actions** CI (`.github/workflows/tests.yml`)
- ✅ Lockfile and dependency consistency guard
- ✅ Governance kernel integrity and drift validation
- ✅ Adversarial contradiction simulations
- ✅ Coverage reporting to Codecov (85 %)

## Repository Layout

| Path | Purpose |
|------|----------|
| `tessrax/` | Core engines (Memory, Metabolism, Governance, Trust) |
| `examples/` | Demonstrations and CLI simulations |
| `docs/` | Architecture and protocol documentation |
| `dashboard/` | Streamlit/React visualization layer |
| `automation/` | CI helpers, lockfile scripts |
| `tests/` | Pytest suite |
| `docker/` | Multi-node FastAPI cluster deployment |
| `tessrax_truth_api/` | REST interface for governance receipts |

## AI Skills Prompt Toolkit

The `ai_skills` module provides deterministic prompt templates and evaluation tools.

### Example: Render a Socratic Debugger

```bash
python -m ai_skills.prompting.cli render   --template socratic_debugger   --task "Add numbers"   --context "3 and 5"
```

### Validate a Guess

```bash
python -m ai_skills.prompting.cli score --guess "test" --truth "test"
```

Full checklist:
```bash
pytest tests/ -v
python -m ai_skills.prompting.cli render --template socratic_debugger --task "test" --context "test"
python -m ai_skills.prompting.cli score --guess "test" --truth "test"
```

## Why Tessrax Exists

Modern AI systems are non-deterministic and often involve multiple agents, so conflicting outputs are common. Tessrax records those conflicts, applies reconciliation rules, and preserves the results in a verifiable ledger.

## Dockerized Cluster & Dashboard

A three-node demo cluster ships with `docker/docker-compose.yml`:

- `governance-node-a` / `governance-node-b`: FastAPI services that replicate ledgers to a moto-backed S3 bucket
- `tessrax-etl-worker`: ETL pipeline that ingests telemetry into the cluster

```bash
cd docker
docker compose up --build
```

The dashboard is available at `http://localhost:8000/dashboard`.

## Stripe Test-Mode Billing

The monetization API exposes `/billing/checkout` and `/billing/subscribe`, using a Stripe test gateway for subscription flow demonstrations.

## Citation

> Vetos, J. (2025). *Tessrax: A Framework for AI Governance and Contradiction Metabolism.* GitHub.
> [https://github.com/joshuavetos/Tessrax](https://github.com/joshuavetos/Tessrax)

## Community & Contributions

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
