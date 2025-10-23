# Getting Started with Tessrax

This guide walks new contributors through environment setup, running examples, and validating governance receipts.

## Prerequisites

- Python 3.10 or newer
- Git and make (optional but recommended)
- Node.js 18+ for dashboard development (optional)
- OpenSSL 1.1+ for local signing demos

## Repository Setup

```bash
# Clone and enter the repository
git clone https://github.com/YOUR_ORG/Tessrax.git
cd Tessrax

# Create an isolated environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install Tessrax in editable mode with development extras
pip install -e .[dev]
```

## Configuration

1. Copy `.env.example` to `.env` if your deployment requires secrets.
2. Configure connectors via `config/connectors.yml` (create the file if absent).
3. Adjust ledger storage paths in `tessrax/ledger.py` when persisting to disk.

## Running Automated Examples

```bash
python examples/minimal_detect.py
python examples/governance_demo.py
python examples/dashboard_snapshot.py
```

Each script emits deterministic demo output to make debugging straightforward.

## Launching the Dashboard (Optional)

```bash
npm install --prefix dashboard
npm run dev --prefix dashboard
```

By default the dev server binds to `http://localhost:5173`. Configure API base URLs in `dashboard/.env`.

## Running Tests

```bash
pytest
```

For coverage reports:

```bash
pytest --cov=tessrax --cov-report=term-missing
```

## Troubleshooting

- **Dependency Conflicts** – Recreate the virtual environment and reinstall with `pip install -e .[dev]`.
- **Missing Dashboard Build Tools** – Install Node.js and pnpm/yarn if you prefer alternative package managers.
- **Ledger Verification Errors** – Run `python -m tessrax.ledger verify path/to/ledger.jsonl` to pinpoint the invalid receipt.
- **Security Concerns** – Follow the disclosure process in [SECURITY.md](../SECURITY.md).
