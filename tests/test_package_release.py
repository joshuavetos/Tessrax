import json
import subprocess
import sys
from pathlib import Path


ARTIFACTS = [
    Path('out') / 'Tessrax-v19-release.tar.gz',
    Path('out') / 'Tessrax-v19-release.whl',
    Path('out') / 'release_packager_receipt.json',
]


def test_package_release_generates_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    for artifact in ARTIFACTS:
        artifact_path = repo_root / artifact
        if artifact_path.exists():
            artifact_path.unlink()
    subprocess.run(
        [sys.executable, 'scripts/package_release.py'],
        cwd=repo_root,
        check=True,
    )
    for artifact in ARTIFACTS[:-1]:
        artifact_path = repo_root / artifact
        assert artifact_path.exists()
        assert artifact_path.stat().st_size > 0
    receipt_data = json.loads((repo_root / ARTIFACTS[-1]).read_text())
    assert receipt_data['integrity'] >= 0.95
    assert receipt_data['legitimacy'] >= 0.9
    assert receipt_data['signature']
    assert receipt_data['status'] == 'pass'
