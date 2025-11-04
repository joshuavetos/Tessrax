# Tessrax SDK (Governed)

The Tessrax SDK provides a lightweight, dependency-free Python client for
interacting with governed Tessrax services.  All operations align with
`AEP-001`, `POST-AUDIT-001`, `RVC-001`, and `EAC-001`.

## Usage

```python
from sdk.tessrax_client import submit_claim, verify_receipt

submit_claim(
    subject="model-a",
    metric="precision",
    value=0.92,
    timestamp="2024-05-04T10:00:00Z",
    source="sdk-example"
)
```

The helpers automatically retry network requests (up to 3 attempts with a
5-second timeout) and raise typed exceptions:

- `ConnectionError` — network failures after retries
- `SchemaError` — invalid JSON responses
- `ProtocolError` — governance contract violations

Call `verify_receipt(receipt_id)` to confirm a ledger entry exists.
