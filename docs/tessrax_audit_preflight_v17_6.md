# Tessrax Repository Audit Pre-Flight Requirements v17.6

> **Governance scope:** Codex deployments engaging with Tessrax repositories MUST satisfy these pre-flight controls before any automated audit, branch creation, or PR workflow executes. Requirements align with AEP-001, DLK-001, and POST-AUDIT-001 mandates.

## 1. Secure Authenticated GitHub Context

- Run inside a CI or Codex environment that provides an authenticated GitHub token (`GH_TOKEN`) with `repo:write` and `pull_request` scopes.
- NEVER bake credentials into scripts. Store `GH_TOKEN`, `GIT_COMMIT_USER`, and `GIT_COMMIT_EMAIL` in your CI secret manager.
- Set the Git identity for audit automation:

  ```bash
  git config --global user.name "Tessrax Automation"
  git config --global user.email "ci@tessrax.com"
  ```

- Verify repository access and session context:

  ```bash
  gh auth status
  git ls-remote origin
  ```

## 2. Sanctioned Bulk-Inspection Tooling

- AEP-001 forbids unsanctioned bulk listings. Use only approved manifest utilities:
  - `automation/manifest_export.py`

    ```bash
    python automation/manifest_export.py --out audits/repo_manifest_v17_6.json
    ```

  - `git ls-tree -r --name-only HEAD > audits/repo_filelist.txt`
  - A repo-root `os.walk()` invocation inside a governed Python script.
- Do **NOT** run raw `rm`, `mv`, or arbitrary file writers during inspection. Manifest outputs become the evidence bundle for audit receipts.

## 3. Double-Lock (DLK) Verification Pipeline Activation

- DLK-001 requires both locks engaged before audits commence.
- Confirm the governance kernel is active and logging DLK events:

  ```bash
  python core/governance_kernel.py --verify-dlk
  ```

- Expected validation output: `[DLK] Pre-lock ✔  Post-lock ✔  Integrity ≥ 0.9`.

## 4. Execute the Audit Under Verified Conditions

- After DLK verification succeeds, launch the audit workflow:

  ```bash
  python automation/run_repo_audit.py \
      --manifest audits/repo_manifest_v17_6.json \
      --dlk \
      --export audits/repo_coherence_audit_v17_6.json
  ```

- This guarantees manifest alignment and DLK-backed receipts for audit artifacts.

## 5. Ledger Receipts and Compliance Log

- Append the audit artifacts to the Tessrax ledger and verify integrity:

  ```bash
  python core/ledger/ledger.py append audits/repo_coherence_audit_v17_6.json
  python core/ledger/ledger.py verify --latest
  ```

- Approval requires `Integrity ≥ 0.95` prior to merge.

---

**Audit Receipt Template (DLK-VERIFIED)**

```json
{
  "auditor": "Tessrax Governance Kernel v16",
  "clauses": ["AEP-001", "POST-AUDIT-001", "RVC-001", "EAC-001"],
  "timestamp": "<UTC ISO8601>",
  "runtime_info": {
    "environment": "<ci-system>",
    "git_ref": "<commit-sha>"
  },
  "status": "pre-flight-complete",
  "integrity_score": 0.95,
  "signature": "<sha256 or ed25519>"
}
```

> **Note:** If any step fails, halt and emit a `TESS MODE BLOCK` with remediation guidance. Never proceed with a partial or unverified audit.
