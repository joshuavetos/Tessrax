# Tessrax Meta Launcher

This automation package prepares the Tessrax launch campaign using the audited messaging assets.

## Governance Inputs
- `governance_receipt.json` — audit record signed by Tessrax Governance Kernel v14.
- `config.json` — references the governance receipt and channel metadata derived from the audit platform analysis.
- `post_templates/` — per-channel markdown templates with approved language and visual placeholders.

The scheduler computes a SHA256 hash of `governance_receipt.json` for each run. The hash is embedded in every log header and in the ledger summary to demonstrate provenance.

## Usage
```bash
python automation/meta_launcher/scheduler.py preview
python automation/meta_launcher/scheduler.py run --dry-run
python automation/meta_launcher/scheduler.py run --live  # requires manual approval and credentials
```

- `preview` prints a chronological plan with launch windows and titles.
- `run --dry-run` simulates the campaign, writes sandbox logs, and appends the ledger.
- `run --live` enables live mode. Mock clients are provided as placeholders for approved integrations; never store credentials in the repository.

Logs are written to `automation/meta_launcher/logs/`. Each run appends a JSON line to `ledger.log` summarizing the actions taken, ensuring transparent auditability.

## Safety and Auditability
- Dry-run mode is the default; live mode must be explicitly requested with `--live` and is still sandboxed until integrations are authorized.
- All simulated posts are timestamped and logged with the governance hash for traceability.
- The ledger ensures every run is recorded, satisfying governance kernel requirements.

Refer back to `governance_receipt.json` for the full list of validated claims, unfalsifiable flags, and the launch checklist endorsed by the audit.
