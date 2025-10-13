# Contributing to Tessrax

We’re thrilled you want to collaborate. Tessrax is a governance engine that grows stronger with every thoughtful contribution. This guide keeps our contributions coherent, auditable, and fast to review.

---

## 🔧 Quick Start (Dev Setup)

1) **Clone**
```bash
git clone https://github.com/joshuavetos/tessrax.git
cd tessrax

	2.	Python 3.11+ (recommended). Create a venv and install dev deps:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"        # installs tessrax + dev extras
# Optional: pre-commit hooks for lint/format
pre-commit install

	3.	Run test suite

pytest -q
# Coverage gate: 85% minimum for changed code
pytest --cov=tessrax --cov-report=term-missing

	4.	Type-check & lint (CI parity)

mypy tessrax
ruff check .
ruff format .
bandit -r tessrax      # security scan
pip-audit              # dependency vuln audit

	5.	Run demo / UI

python tessrax/demo_flow.py
python tessrax/current.py     # launches API + dashboard


⸻

🌿 Branching Model
   •   main — protected; stable, tagged releases only.
   •   develop — integration branch for completed features.
   •   feature/<slug> — scoped feature work (small, iterative PRs).
   •   fix/<slug> — scoped bugfix work.
   •   release/<version> — pre-release hardening.

Keep branches short-lived; rebase onto develop before PR.

⸻

📝 Commit Convention

Use Conventional Commits for traceable history and auto-changelogs:

feat(core): add scar severity indexing
fix(rigor/charity): resolve scope partition bug
docs(readme): clarify federation notes
refactor(ledger): simplify merkle root calc
test(governance): add quorum threshold cases
perf(semantic): cache normalized entities
chore(ci): pin mypy version

Include an imperative subject, optional scope, and concise body with rationale, links to issues, and BREAKING CHANGE notes if relevant.

⸻

🔁 Pull Requests

Target: develop

Checklist (required):
   •   Tests added/updated; suite passes locally (pytest -q)
   •   Lint/format/type-check pass (ruff, mypy, pre-commit)
   •   Coverage ≥ 85% on changed lines (CI enforces)
   •   Docs updated (module docstrings and /docs where applicable)
   •   No secrets or PII in code, tests, or fixtures
   •   For governance-sensitive changes: updated charter/ or compliance/ as needed and included a short governance note in PR description
   •   Ledger-affecting logic includes receipt/merkle test vectors

How to PR
	1.	Fork → create feature/<slug>
	2.	Commit using Conventional Commits
	3.	Rebase on develop (no merge commits if possible)
	4.	Open PR to develop with:
      •   What changed & why
      •   Risks / rollback plan
      •   Any migration steps
      •   Screenshots (UI) or sample receipts (ledger)

Reviews
   •   One maintainer approval minimum; two for governance/ledger-critical areas.
   •   “Changes requested” pauses merge until addressed.
   •   CI must be fully green.

⸻

🧪 Testing Standards
   •   Unit tests (tests/unit/): pure logic, fast (<200ms each)
   •   Integration (tests/integration/): API, ledger, receipts, sandbox
   •   Rigor Layer (tests/rigor/): Hierarchy, Telos, Charity, Observer, Invariants, Multi-Scale
   •   Golden vectors for receipts/merkle under tests/data/
   •   Use pytest fixtures; avoid network access; use temporary FS.
   •   Deterministic seeds for stochastic pieces.

Coverage Policy
   •   New modules: ≥90% lines; changed lines: ≥85%.
   •   Core crypto/ledger paths: 100% on happy-path + failure-path.

⸻

🧭 Code Standards
   •   Python 3.11+; PEP 8/257 aligned via ruff + ruff format
   •   Type hints required (mypy --strict clean for new/changed modules)
   •   Public functions must have docstrings with brief examples
   •   Avoid global state; prefer pure functions and explicit deps
   •   Keep functions short; extract helpers rather than nesting deeply
   •   Errors: raise precise exceptions; include remediation hints

Modules of Interest
   •   tessrax/core/ — contradiction detection, metabolism orchestration
   •   tessrax/rigor/ — integrity band modules (Hierarchy, Telos, Charity, Observer, Invariants, Multi-Scale)
   •   tessrax/ledger/ — append-only JSONL + Merkle root anchoring
   •   tessrax/receipts/ — tamper-evident receipts & signatures
   •   tessrax/dashboard/ — Flask/D3 audit views

⸻

🔐 Security & Compliance
   •   No secrets/PII in repo, tests, or logs. Use .env locally and mock secrets in tests.
   •   Run bandit, pip-audit before PR.
   •   Ledger files may contain hashes and non-sensitive metadata only.
   •   Changes to charter/quorum/revocation require a governance note and additional reviewer.

Responsible Disclosure (Security)
   •   Email: security@tessrax.example (placeholder)
   •   Do not open public issues for 0-days; we’ll coordinate a fix and release.

⸻

🧭 Design Guardrails (Rigor Layer)

Before you mark any conflict a “contradiction,” apply:
	1.	Principle of Charity — normalize terms, align scopes, check definitions.
	2.	Observer Relativity — verify it’s not an artifact of models/instruments.
	3.	Equilibria & Invariants — allow lawful static truths and conserved relations.
	4.	Multi-Scale — translate to shared scale; avoid category errors.
	5.	Hierarchy of Differences — benign variety ≠ contradiction; escalate only when required.

These are enforced in code under tessrax/rigor/ and must be respected by new features.

⸻

🗂 Documentation
   •   Update inline docstrings and /docs/ for any user-facing change.
   •   For endpoints/UI, include request/response examples and screenshots/GIFs.
   •   Governance-affecting changes must update the changelog and charter notes.

⸻

🚢 Release Process (Maintainers)
	1.	Cut release/<version> from develop
	2.	Freeze features; bump version; update CHANGELOG.md
	3.	Run full CI, security scans, and e2e smoke
	4.	Merge release/<version> → main (tag), then back-merge main → develop

⸻

🤝 Contribution Types
   •   Code — engines, rigor modules, sandbox hardening, performance
   •   Docs — guides, architecture notes, tutorials
   •   Tests — gold vectors, fuzzers, integration scenarios
   •   Ops — CI improvements, build reproducibility, container tweaks
   •   Community — triage, design proposals, governance discussions

Contributors are credited in release notes and /docs/contributors.md.

⸻

🧾 Issue Reports & Feature Requests
   •   Use templates:
      •   Bug — steps to reproduce, expected vs actual, logs (redacted), env
      •   Feature — use case, proposed API, risks, alternatives
   •   Tag with area/* (core, ledger, rigor, dashboard, docs)

⸻

🗄️ Suggested .gitignore

Save as .gitignore in repo root.

# Python
__pycache__/
*.py[cod]
*.so
*.pyd
*.pyo
*.db
.mypy_cache/
.pytest_cache/
.coverage
htmlcov/

# Packaging / build
build/
dist/
*.egg-info/

# Virtual environments
.env
.venv/
venv/

# Editors/IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Node / dashboard
node_modules/
dashboard/.cache/
dashboard/.parcel-cache/
dashboard/dist/

# Logs & temp
logs/
*.log
tmp/
.cache/

# Tessrax artifacts
artifacts/
data/
ledgers/
*.jsonl
receipts/
anchors/
secrets.*
.env.*

# Local test fixtures (never commit real data)
tests/fixtures/local/


⸻

📣 Code of Conduct

We follow a standard open-source Code of Conduct (be respectful; no harassment).
Contact maintainers via issues or email listed in COMMITTERS.md for moderation.

⸻

Thanks!

Tessrax exists to metabolize contradictions into clarity. Your contributions make the organism stronger, more auditable, and more useful for everyone.

