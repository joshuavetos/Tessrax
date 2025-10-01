# Contributing to Tessrax

We’re excited you want to contribute! This project thrives on collaboration.

---

## 🛠 Development Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/joshuavetos/tessrax.git
   cd tessrax

2.	Install dependencies (Python 3.10+ recommended).

3.	Run tests:

pytest tests/

Contribution Guidelines
   •   Branches
      •   main: stable release only.
      •   develop: active development.
      •   feature/*: feature work.
      •   release/*: pre-release stabilization.
   •   Commits
      •   Use clear, descriptive commit messages.
      •   Example: feat: add scar severity indexing.
   •   Pull Requests
      •   Fork → feature branch → PR into develop.
      •   Ensure all tests pass.
      •   Add/update documentation when relevant.

⸻

📦 Code Standards
   •   Use PEP8 style guidelines.
   •   Include docstrings and type hints.
   •   Write unit tests for new features.

⸻

🧩 Contribution Types
   •   Code: engine improvements, new modules, bug fixes.
   •   Docs: improve guides, fix typos, add tutorials.
   •   Examples: new demos, integrations.
   •   Community: triage issues, suggest features, security reports.

⸻

🙌 Recognition

All contributors will be credited in release notes and documentation.

---

### 📌 `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.db

# Virtual environments
.env
.venv

# IDEs
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Tessrax ledgers
*.jsonl
