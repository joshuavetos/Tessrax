{
  "CE-MOD-66": {
    "name": "Tessrax Enhanced Conflict Graph",
    "version": "3.0",
    "author": "Joshua Vetos / Claude (Anthropic)",
    "type": "Contradiction Engine",
    "status": "stable",
    "function": "Graph-based contradiction auditor that ingests structured contradiction ledgers (v4 schema) and maps them into a weighted semantic graph. Detects logical, numeric, temporal, and categorical inconsistencies and logs all findings to the governance kernel.",
    "features": [
      "Semantic similarity scoring (SentenceTransformer)",
      "Sentiment and entity extraction via Transformers",
      "Unit-aware numeric comparison (pint)",
      "Temporal contradiction parsing (parsedatetime)",
      "Graph clustering and centrality metrics (networkx)",
      "Governance-kernel event hooks (AGENT_ANALYSIS_REPORT)",
      "Interactive visualization layer (Plotly optional)"
    ],
    "inputs": {
      "ledger": "/core/data/ledger/contradiction_ledger_v4.json",
      "config": "/core/config/ce-mod-66.yaml"
    },
    "outputs": {
      "graph": "networkx.Graph",
      "ledger": "/core/data/logs/contradiction_ledger_log.jsonl",
      "summary": "Top contradiction nodes with weighted scores",
      "checksum": "auto-SHA256"
    },
    "linked_scarpath": [
      "ς_truth_vs_appearance",
      "ς_metric_drift",
      "ς_promised_vs_proven"
    ],
    "governance_hooks": {
      "kernel_event": "AGENT_ANALYSIS_REPORT",
      "audit_pipeline": "governance_kernel.py",
      "visualization": "visualize_scaffolding.py"
    },
    "origin_file": "conflict_graph.py",
    "signature": "canonized 2025-10-08 under SIG-LOCK-001"
  }
}