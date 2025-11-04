# Human Feedback API (DLK-Verified)

The governed Human Feedback API accepts corrective inputs from authenticated
operators.  All submissions are processed under Tessrax Governance Kernel v16
and recorded with the clauses `AEP-001`, `POST-AUDIT-001`, `RVC-001`, and
`EAC-001`.

## Endpoint

`POST /feedback/`

### Request Body

```json
{
  "claim_id": "string",
  "correction": "string",
  "user_id": "string",
  "signature": "optional string",
  "metadata": {"optional": "object"}
}
```

### Governance Features

- IP addresses are anonymised via SHA-256 hashing for debiasing audits.
- User-agent metadata is retained to improve provenance analytics.
- Optional digital signatures can be supplied by responders for independent
  provenance verification.
- Rate limiting (3 submissions per minute per user/IP hash) prevents
  governance spam without blocking legitimate corrections.

### Response

```json
{
  "receipt_id": "hex string",
  "status": "recorded",
  "integrity_score": 0.95,
  "timestamp": "ISO-8601"
}
```

Each response corresponds to a ledger entry with `event_type` set to
`HUMAN_FEEDBACK_RECEIPT` and is DLK-verified.
