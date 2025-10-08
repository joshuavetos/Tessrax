| Threat | Likelihood | Impact | Mitigation |
|--------|-------------|---------|-------------|
| Ledger tampering | Medium | Critical | Hash chaining + Merkle anchoring |
| Sandbox escape | Medium | High | RestrictedPython + container isolation |
| Key compromise | Low | Critical | Rotation + revocation list |
| Consensus drift | Medium | Medium | Federation self-healing |
| Data exfiltration | Low | High | /data/ sandbox + no outbound network |

sentence-transformers==3.0.1
transformers==4.44.2
networkx==3.2.1
pint==0.24.1
parsedatetime==2.6
PyNaCl==1.5.0
RestrictedPython==7.2
prometheus_client==0.20.0