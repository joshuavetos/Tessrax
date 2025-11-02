import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path("tessrax_truth_api/services/stripe_gateway.py").resolve()
_SPEC = importlib.util.spec_from_file_location("stripe_gateway", _MODULE_PATH)
assert _SPEC and _SPEC.loader  # runtime guard
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
StripeGateway = _MODULE.StripeGateway


def test_stripe_gateway_mock_checkout() -> None:
    gateway = StripeGateway(api_key="sk_test_mocked_key", mock=True)
    session = gateway.create_checkout_session(
        amount_cents=4900,
        currency="usd",
        tier="starter",
        success_url="https://example.com/success",
        cancel_url="https://example.com/cancel",
    )
    assert session.id.startswith("cs_test_")
    assert session.amount_total == 4900
    assert session.currency == "usd"
