````markdown name=tessrax_security.md url=https://github.com/joshuavetos/Tessrax/blob/38bef2b0a12af67e35504eee7c6433e8a294db12/tessrax_security.md
# Tessrax Security Policy

## Supported Versions

| Version   | Status            | Notes         |
|-----------|-------------------|---------------|
| >= 2.2    | Supported         | Active support, receives security updates |
| < 2.2     | Unsupported       | Demo only; no patches or issue triage     |

## Vulnerability Disclosure

If you discover a security vulnerability, please report it directly and privately. Use either:
- Email: [maintainer-email@example.com]
- GitHub Security Advisories

**Do not disclose vulnerabilities publicly until a fix is released.**  
We will acknowledge receipt within 2 business days and aim to provide a remediation timeline within 7 business days.

## Security Commitments

- **Append-Only Logging:**  
  All logs are designed to be append-only, preventing unauthorized tampering or deletion of history.

- **Verification & Integrity Tools:**  
  Tools are provided to verify log integrity and detect any data drift or corruption.

- **Transparency:**  
  Security-relevant changes, mitigations, and incident responses are documented and communicated promptly.

- **Future Enhancements:**  
  Plans exist to provide optional blockchain-based anchoring for logs to further guarantee immutability and auditability.

## Current Limitations & Risks

- **Storage Backend:**  
  Data is currently stored in local JSONL (JSON Lines) format. This approach is susceptible to race conditions and potential data corruption if multiple write operations occur concurrently.

- **Multi-User Environments:**  
  Tessrax is not yet hardened for multi-user or production deployments. There is no built-in access control or concurrent write protection.

- **Encryption & Access Control:**  
  Data at rest and in transit is not encrypted by default. There is currently no integrated authentication or authorization.

- **Audit Logging:**  
  Security-relevant events (e.g., failed verification, unauthorized access attempts) may not be fully logged or monitored.

## Recommendations for Secure Use

- **Single-User Only:**  
  Deploy Tessrax only in single-user, trusted environments until further hardening is complete.

- **Backups:**  
  Regularly back up your log files and test recovery procedures.

- **Monitor for Updates:**  
  Stay informed about security updates and upgrade promptly.

- **Contribute Feedback:**  
  Security feedback and contributions are welcome—please open an advisory or contact the maintainers.

## Roadmap

- Harden storage backend for concurrent and multi-user safety
- Add built-in authentication and role-based access control
- Enable encryption for data at rest and in transit
- Implement blockchain-based log anchoring (optional)
- Expand automated security and integrity testing

---

_Last updated: 2025-10-05_
````