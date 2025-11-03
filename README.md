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

It detects logical and semantic conflicts across data, models, or agent outputs—then metabolizes those conflicts into verified clarity through formal governance protocols.

Every reconciliation becomes a **signed, hash-linked ledger receipt**, proving not just that a decision was made, but *how and why*.

> ⚖️ **Contradiction is not a bug — it’s a signal.**  
> Tessrax governs it, audits it, and transforms it into clarity.

Use Tessrax to:
- Detect and reconcile contradictions in AI or multi-agent systems  
- Enforce epistemic integrity across distributed or self-auditing environments  
- Generate cryptographically verifiable receipts of governance decisions  

---

## 🔩 Core Architecture

| Layer | Purpose |
|-------|----------|
| **Memory Engine** | Stores all claims, contradictions, and reconciliation logs |
| **Metabolism Engine** | Converts contradictions into structured clarity statements |
| **Governance Kernel** | Executes policy and quorum logic over uncertain or conflicting data |
| **Trust Engine** | Signs, verifies, and anchors every ledger entry cryptographically |
| **Dashboard** | Visualizes contradictions, resolutions, and epistemic health metrics in real time |

<details>
<summary>📊 View Architecture Diagram</summary>

![Tessrax System Diagram](docs/assets/architecture_overview.svg)
</details>

---

## 🚀 Quick Start

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

📄 For full setup and advanced configuration, see [`docs/getting_started.md`](docs/getting_started.md).

---

## 🧾 Governance Receipts

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

Receipts are tamper-evident, Merkle-linked, and independently verifiable.

---

## 🧪 Demonstrations

| Script | Description |
|---------|-------------|
| `examples/minimal_detect.py` | Loads sample claims and flags contradictions |
| `examples/governance_demo.py` | Runs a full governance cycle and appends ledger receipts |
| `examples/dashboard_snapshot.py` | Produces a Streamlit-ready visualization snapshot |

```bash
python examples/<script>.py
```

---

## 🔬 Runtime Integrity

| Check | Method |
|--------|--------|
| ✅ Python | 3.11.x (enforced by CI) |
| 🔒 Lockfiles | `requirements-lock.txt` auto-generated via `automation/regenerate_lock.sh` |
| 🧠 Imports | Absolute (`tessrax.*`) enforced via `.flake8` |
| 🧪 Tests | `pytest -q` |
| 🧬 Self-Test | `python -m tessrax.metabolism.reconcile --self-test` |

All test results and receipts are mirrored in the **audit ledger** under `ledger/ledger.jsonl`.

---

## 🗂️ Field Evidence Archive (2025-10-28)

Tessrax ships with a **Field Evidence Archive**, a JSONL dataset of real and simulated contradiction audits gathered during 2023–2025. It powers simulation benchmarks and compliance metrics inside the governance kernel.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Stable identifier |
| `category` | `str` | Domain (e.g. `supply_chain`, `safety`) |
| `source_type` | `str` | Provenance (`field_report`, `simulation`, etc.) |
| `summary` | `str` | Narrative synopsis |
| `alignment` | `dict` | Policy reference, status, and score |

Located at `tessrax/data/evidence/field_evidence_archive_2025-10-28.jsonl`.

---

## 🧱 Verification & Continuous Integration

Tessrax performs a full reproducibility audit on every commit:

- ✅ **GitHub Actions** CI (`.github/workflows/tests.yml`)
- ✅ Lockfile and dependency consistency guard
- ✅ Governance kernel integrity and drift validation
- ✅ Adversarial contradiction simulations
- ✅ Coverage reporting to Codecov (85 %)

---

## 📦 Repository Layout

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

---

## 🧩 AI Skills Prompt Toolkit

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

---

## 🧭 Why Tessrax Exists

Modern AI systems are non-deterministic, multi-agent, and epistemically unstable. Contradictions are inevitable. Tessrax turns that inevitability into architecture:

- It doesn’t suppress conflict — it **resolves** and **records** it.  
- It doesn’t trust blindly — it **signs, verifies, and proves**.  
- It metabolizes contradiction into **governance clarity**.

> **Tessrax:** turning contradiction into cognition, and cognition into proof.

---

## 🧰 Dockerized Cluster & Dashboard

A three-node demo cluster ships with `docker/docker-compose.yml`:

- `governance-node-a` / `governance-node-b` — FastAPI services replicating ledgers to a moto-backed S3 bucket  
- `tessrax-etl-worker` — ETL pipeline ingesting telemetry into the cluster

```bash
cd docker
docker compose up --build
```

Visit `http://localhost:8000/dashboard` for the React-based control tower.

---

## 💳 Stripe Test-Mode Billing

The monetization API exposes `/billing/checkout` and `/billing/subscribe`, using a Stripe test gateway for sandboxed subscription flow demonstrations.

---

## 🔬 Citation

> Vetos, J. (2025). *Tessrax: A Framework for AI Governance and Contradiction Metabolism.* GitHub.  
> [https://github.com/joshuavetos/Tessrax](https://github.com/joshuavetos/Tessrax)

---

## 🛡️ Community & Contributions

- 📜 [Code of Conduct](CODE_OF_CONDUCT.md)  
- 🧪 [Contributing Guide](CONTRIBUTING.md)  
- 🔐 [Security Policy](SECURITY.md)

---

## 🧠 License

Released under the MIT License. See [LICENSE](LICENSE) for details.
