import pytest
from core.contradiction_engine import safe_match, TimeoutException

def test_regex_timeout():
    evil_pattern = r"^(a+)+$"
    with pytest.raises(RuntimeError):
        safe_match(evil_pattern, "a"*10000, timeout=0.1)

def test_regex_safe_pattern():
    assert safe_match(r"\bnow\b", "now or later")
    
    if __name__ == "__main__":
    try:
        test_regex_timeout()
        test_regex_safe_pattern()
        print("✅ Regex safety tests passed.")
    except Exception as e:
        print("❌ Regex safety test failed:", e)