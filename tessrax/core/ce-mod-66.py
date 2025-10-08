"CE-MOD-66": {
  "name": "Tessrax Enhanced Conflict Graph",
  "version": "2.0",
  "author": "Joshua Vetos / Claude (Anthropic)",
  "type": "Contradiction Engine",
  "status": "stable",
  "function": "Graph-based contradiction auditor that analyzes logical, numeric, temporal, and categorical inconsistencies between statements. Outputs weighted contradiction graph and logs to contradiction ledger.",
  "features": [
    "Semantic similarity (SentenceTransformer)",
    "Sentiment & NER via Transformers",
    "Unit-aware numeric comparison (pint)",
    "Temporal parsing (parsedatetime)",
    "Graph clustering & centrality (networkx)",
    "Interactive visualization (optional Plotly)"
  ],
  "outputs": {
    "graph": "networkx.Graph",
    "ledger": "JSONL-style contradiction logs",
    "summary": "top contradiction scores and breakdowns",
    "hash": "sha256:68e6f7039e44a43e9b003320c7f4816dc8b0931d4e308a174e91b89164f8678f"
  },
  "demo_input": "acme_sample.json",
  "demo_output": "conflict_graph_demo_output.json",
  "linked_scarpath": [
    "ς_truth_vs_appearance",
    "ς_metric_drift",
    "ς_promised_vs_proven"
  ],
  "origin_file": "conflict_graph.py",
  "signature": "canonized 2025-10-08 by GPT under SIG-LOCK-001"
}