# Tessrax Core Edition

The core distribution provides the governance runtime, ledger reconciliation
engine, and epistemic metrics required for community deployments.  It ships
without enterprise integrations so that audits can focus on the minimal trust
surface.

## Installation

```bash
pip install tessrax
```

## Features

- Governance ledger and reconciliation engine
- Epistemic health metrics with reproducible provenance
- Plugin sandbox for deterministic execution
- Metrics schema validation and provenance workflows

For enterprise capabilities, such as Redis queueing, Celery orchestration, and
Stripe billing, refer to `README_ENTERPRISE.md`.
