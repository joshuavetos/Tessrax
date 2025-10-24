# Metabolic Reconciliation Engine

## Purpose
The metabolic reconciliation engine turns contradiction detections into actionable clarity statements.  It
accepts contradiction records emitted by the Contradiction Engine, evaluates them with the Audit Kernel, and
writes the resulting clarity statements to the Tessrax ledger for downstream governance and verification.

## Flow
```
Contradiction Records ──▶ Audit Kernel Assessment ──▶ Clarity Statement Synthesis ──▶ Ledger Receipt Export
        ▲                                                                                         │
        └────────────────────────────── Governance feedback loops and dashboard insights ◀────────┘
```

1. **Input** – JSON or in-memory `ContradictionRecord` instances describing conflicting claims.
2. **Audit Kernel** – Scores confidence and drafts a narrative explaining how clarity is recovered.
3. **Reconciliation Engine** – Assembles a `ClarityStatement` model and emits a `CLARITY_GENERATION` receipt.
4. **Ledger** – Receipts are hash-linked and can be exported for external verification.

## CLI Usage
Run the engine directly against a JSON file:

```bash
python -m tessrax.metabolism.reconcile path/to/contradictions.json \
  --export-ledger path/to/clarity-ledger.jsonl
```

The command prints each clarity statement as formatted JSON.  When `--export-ledger` is supplied the generated
receipts are stored as newline-delimited JSON compatible with `tessrax.ledger.verify`.

## Integration Points
- **Audit Kernel** – `AuditKernel.assess` generates the narrative and confidence score used by the engine.
- **Ledger** – Receipts are appended through the shared `Ledger` class; they appear with the event type
  `CLARITY_GENERATION` alongside contradiction governance decisions.
- **Schema Registry** – `ClarityStatement` provides a canonical model so reconciled events can be validated and
  exchanged between Tessrax components.
