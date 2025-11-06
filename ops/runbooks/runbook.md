Tessrax Governance Runbook (v19.6)
==================================

All operational alerts map directly to receipts and recovery procedures.

Alert → Action Mapping
----------------------

### HighReceiptLatency

* **Source:** Prometheus HighReceiptLatency
* **Action:** Check `/metrics` for `queue_depth_ratio`. If >0.7, scale metabolism worker pods by +1.
* **Receipt linkage:** Inspect last `SHEDDING_RECEIPT` entries in ledger.

### QueueDepthHigh

* **Source:** Prometheus QueueDepthHigh
* **Action:** Increase batch size threshold in `governance_kernel.yaml` or deploy one more replica.
* **Expected Resolution:** Receipt latency should drop within 5m.

### IntegrityDriftDetected

* **Action:** Query `/api/status?window=1000` to identify divergent entries.
* **If sustained:** Pause federation sync (`federation.orchestrator.pause()`), run `verify_chain()`.

### ByzantineRootMismatch

* **Action:** Trigger consensus audit via `federation/orchestrator.py --verify`.
* **Record:** Append audit output as `POLICY_VIOLATION` receipt with Merkle root delta.

### KeyRotationOverdue

* **Action:** Execute `python -m tessrax.core.key_vault rotate`.
* **Validation:** Confirm `KEY_ROTATION` receipt in ledger with status: `PASS`.

Emergency Procedures
--------------------

### Total Node Failure

1. Restore last checkpoint (`.checkpoint.json`).
2. Run ledger replay:
   ```bash
   python -m tessrax.recover --from-checkpoint .checkpoint.json
   ```
3. Verify with `verify_chain()`.
4. Append recovery receipt to ledger.

### Governance Cascade Loop

1. Issue `governance.breaker.enable()`.
2. Review latest `CYCLE_DETECTED` receipts.
3. Resume after human override with signed `POLICY_OVERRIDE` receipt.

All actions must end with an appended receipt referencing this runbook version (`RUNBOOK_V19_6`).

(Alert definitions and runbook fully mappable to the Prometheus/Grafana setup; drop under `/ops/` and link in deployment manifest.)
