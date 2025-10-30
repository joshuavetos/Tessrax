# Tessrax Enterprise Edition

Enterprise deployments extend the core governance runtime with queueing,
workflow, and billing integrations.

## Installation

```bash
pip install tessrax[enterprise]
```

## Included Integrations

- **Redis** for durable event storage and cache priming
- **Celery** workers for asynchronous contradiction reconciliation
- **Stripe** billing harness for governance credit ledgers

All enterprise extensions are optional and isolated from the core runtime to
simplify audits.  Refer to `docs/enterprise_migration.md` for upgrade guidance
when moving from v15 to v16.
