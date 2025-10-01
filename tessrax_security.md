# Tessrax Security Policy

## Supported Versions
- v2.2 and later: actively maintained
- Earlier versions: unsupported, demo only

## Reporting Vulnerabilities
Please report security issues privately via email or GitHub Security Advisories. Do not disclose publicly until patched.

## Commitments
- Append-only logs: designed to resist tampering
- Verification tools: included to detect drift or corruption
- Future work: optional blockchain anchoring for stronger immutability

## Known Limitations
- Current storage layer is local JSONL; subject to race conditions under concurrent writes
- Not yet hardened for production multi-user environments
