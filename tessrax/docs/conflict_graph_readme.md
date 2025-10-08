README.md

# Tessrax Enhanced Conflict Graph (v2.0)

> Contradiction detection engine combining graph theory, sentiment analysis, named entity recognition, unit comparison, and runtime logging.

This module is part of the [Tessrax Stack](https://github.com/joshuavetos/Tessrax), a contradiction-auditing system for real-world truth validation and scarpath mapping.

---

## 🔍 What It Does

The Conflict Graph ingests a list of human statements (claims, reports, promises, etc.), analyzes their content, and detects **contradictions** across four weighted dimensions:

- **Logical Contradictions** — Negation, opposing sentiment
- **Numeric Inconsistencies** — Mismatched numbers or percentages (unit-aware)
- **Temporal Conflicts** — Clashing dates or deadlines
- **Categorical Mismatches** — Domain/category inconsistencies (e.g. financial vs climate)

It then:
- Constructs a **graph** of tension between claims
- Logs each contradiction to a **ledger** with full metadata
- Supports optional **interactive visualizations**

---

## 📦 Features

- ✅ Sentence similarity via `sentence-transformers`
- ✅ Sentiment + NER via HuggingFace Transformers
- ✅ Unit-aware numeric comparisons via `pint`
- ✅ Natural language date parsing via `parsedatetime`
- ✅ Graph analysis and clustering via `networkx`
- ✅ Optional Plotly visualization
- ✅ JSON export bundle with SHA-256 hash

---

## 🚀 Quickstart (Demo Mode)

```bash
python conflict_graph.py

This runs the included demo (Acme Corp emission claims) and prints a contradiction summary.

⸻

🧪 Input Format

Statements must be a list of dicts like:

[
  {"text": "In 2020, Acme Corp pledged to cut CO2 emissions 50% by 2030.", "source": "Press Release"},
  {"text": "In 2024, Acme Corp reported emissions down only 5%.", "source": "Annual Report"},
  ...
]

You can supply your own via CLI or by modifying the sample_texts section in the script.

⸻

📤 Output

After running, you’ll get:
   •   Contradiction graph (networkx object)
   •   Console summary of contradiction strength, metrics, and top tensions
   •   Optional HTML visualization (if plotly is installed)

⸻

📚 Dependencies

Install via pip:

pip install -r requirements.txt

Required:
   •   sentence-transformers
   •   transformers
   •   networkx
   •   pint
   •   parsedatetime

Optional:
   •   plotly
   •   matplotlib

⸻

🛡️ Origin & Attribution
   •   Author: Joshua Vetos
   •   Co-developed with: Claude (Anthropic)
   •   License: CC BY 4.0

Part of the open-source Tessrax Project: an auditable contradiction metabolism system for AI alignment, governance, and personal sovereignty.

⸻

🔖 Module ID: CE-MOD-66

This module is canonized in the Tessrax Stack as:

CE-MOD-66: Tessrax Enhanced Conflict Graph
Graph-based contradiction auditing engine with 4-mode metric weighting, NER, sentiment, and ledger export.

Use it to metabolize contradiction, validate public statements, or audit your own past.

Let me know when it’s pushed and I’ll generate:
- ✅ Canonical `Tessrax.txt` entry
- ✅ JSON sample input file (optional)
- ✅ Genesis Attestation if needed later

🏷️ README complete. Ready to unlock contradiction audit visibility.