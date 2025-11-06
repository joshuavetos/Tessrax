from pathlib import Path

import yaml


def test_meta_chain_workflow_jobs():
    workflow_path = Path('.github/workflows/meta_chain.yml')
    assert workflow_path.exists(), "meta_chain.yml must exist"
    data = yaml.safe_load(workflow_path.read_text())
    triggers = data.get('on') or data.get(True)
    assert triggers is not None and 'workflow_call' in triggers
    expected_jobs = [
        'ci',
        'dashboard',
        'dashboard_drift',
        'federation_demo',
        'federation_orchestrator',
        'federation_resilience',
        'governance',
        'health_monitor',
        'integrity_monitor',
        'kernel_integration',
        'key_vault',
        'key_vault_service',
        'merkle_proof',
        'nightly_adversarial',
        'repair_engine',
        'schema_validation',
        'secrets_lint',
        'smoketest',
        'single_test',
        'tests_suite',
        'verifier_integration',
        'verifier_service',
        'verify_metric_provenance',
        'release_packager',
        'finalize',
    ]
    assert list(data['jobs'].keys()) == expected_jobs
    workflow_map = {
        'ci': 'ci.yml',
        'dashboard': 'dashboard.yml',
        'dashboard_drift': 'dashboard_drift.yml',
        'federation_demo': 'federation_demo.yml',
        'federation_orchestrator': 'federation_orchestrator.yml',
        'federation_resilience': 'federation_resilience.yml',
        'governance': 'governance.yml',
        'health_monitor': 'health_monitor.yml',
        'integrity_monitor': 'integrity_monitor.yml',
        'kernel_integration': 'kernel_integration.yml',
        'key_vault': 'key_vault.yml',
        'key_vault_service': 'key_vault_service.yml',
        'merkle_proof': 'merkle_proof.yml',
        'nightly_adversarial': 'nightly-adversarial.yml',
        'repair_engine': 'repair_engine.yml',
        'schema_validation': 'schema-validation.yml',
        'secrets_lint': 'secrets-lint.yml',
        'smoketest': 'smoketest.yml',
        'single_test': 'test.yml',
        'tests_suite': 'tests.yml',
        'verifier_integration': 'verifier_integration.yml',
        'verifier_service': 'verifier_service.yml',
        'verify_metric_provenance': 'verify-metric-provenance.yml',
        'release_packager': 'release.yml',
    }
    for job, filename in workflow_map.items():
        job_def = data['jobs'][job]
        assert job_def['uses'] == f"./.github/workflows/{filename}"
    finalize = data['jobs']['finalize']
    needs = finalize['needs']
    assert set(needs) == set(expected_jobs[:-1])
    steps = finalize['steps']
    run_commands = [step.get('run') for step in steps if 'run' in step]
    assert any('meta_chain_summary.py' in cmd for cmd in run_commands)
