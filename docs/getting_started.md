# Getting Started with Tessrax

This guide walks new contributors through environment setup, running examples, and validating governance receipts.

Prerequisites
   •   Python 3.11.x (as enforced by CI)
   •   Git and make (optional but recommended)
   •   Node.js 18+ for dashboard development (optional)
   •   OpenSSL 1.1+ for local signing demos

Repository Setup

# Clone and enter the repository
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax

# Create an isolated environment
python3.11 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install Tessrax in editable mode with development extras
pip install -e .[dev]

Configuration
	1.	Copy .env.example → .env if your deployment requires secrets.
	2.	Configure connectors in config/connectors.yml (create if missing).
	3.	Adjust ledger storage paths in tessrax/core/ledger/ledger.py when persisting receipts locally.

Running Automated Examples

python examples/minimal_detect.py
python examples/governance_demo.py
python examples/dashboard_snapshot.py

Each script emits deterministic output for reproducible debugging.

Launching the Dashboard (Optional)

npm install --prefix dashboard
npm run dev --prefix dashboard

By default, the dev server runs at http://localhost:5173.
Adjust API base URLs in dashboard/.env as needed.

Running Tests

pytest -q

For coverage reports:

pytest --cov=tessrax --cov-report=term-missing

Troubleshooting
   •   Dependency Conflicts – Recreate the virtual environment and reinstall with pip install -e .[dev].
   •   Missing Dashboard Build Tools – Install Node.js and optionally pnpm/yarn for alternate package managers.
   •   Ledger Verification Errors – Run python -m tessrax.core.ledger.verify ledger/ledger.jsonl to locate invalid receipts.
   •   Security Concerns – Follow the disclosure process in SECURITY.md.
