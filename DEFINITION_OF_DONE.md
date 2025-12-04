# Tessrax v33.1 — Definition of Done

## definition_of_done

### architecture
- all endpoints type-checked
- AFCG validation required
- CMD contradiction routes defined
- governance receipts enabled

### infrastructure
- Traefik security headers enabled
- Rate limiting enabled
- TLS store configured
- Docker daemon hardened
- Logs rotate under 50MB

### data
- backups run daily
- retention enforced (14 days)
- restore validated weekly
- ledger replicated air-gapped

### observability
- Prometheus metrics enabled
- Grafana dashboard linked
- error rate < 2% threshold
- latency P95 < 400ms

### security
- trivy scan passes (no critical issues)
- no unencrypted secrets
- kill switch: SAFEPOINT_AFCG_001

### governance
- AFCG: pass
- CMD: contradictions resolved
- GOV: audit trail complete
- ECSL: active & reporting
