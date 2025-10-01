# AI Contradiction Audit System

A tamper-evident logging and governance engine for tracking contradictions in multi-agent AI systems.  
Built in Python, it combines event sourcing, hash-chained logs, and governance rituals into a verifiable audit framework.

---

## ✨ What It Does
- **Contradiction Tracking**: Record and classify contradictions as first-class events.
- **Immutable Ledger**: Append-only JSONL storage with cryptographic chain verification.
- **Scar Registry**: Log contradictions as “scars” with lineage, severity, and status.
- **Governance Claims**: Sign and verify claims with agent identity and timestamp.
- **Continuity Handoffs**: Verifiable chain of custody for system state.
- **Query API**: CLI + REST endpoints to explore scars, claims, and verify chain integrity.

---

## 🔧 Use Cases
- **AI Safety Research**
- **Multi-Agent Debugging**
- **Compliance Auditing**
- **Governance Infrastructure**

---

##  Quick Start
```bash
git clone https://github.com/joshuavetos/Tessrax.git
cd Tessrax
python src/tessrax_engine/engine.py
