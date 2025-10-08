| Threat | Likelihood | Impact | Mitigation |
|--------|-------------|---------|-------------|
| Ledger tampering | Medium | Critical | Hash chaining + Merkle anchoring |
| Sandbox escape | Medium | High | RestrictedPython + container isolation |
| Key compromise | Low | Critical | Rotation + revocation list |
| Consensus drift | Medium | Medium | Federation self-healing |
| Data exfiltration | Low | High | /data/ sandbox + no outbound network |