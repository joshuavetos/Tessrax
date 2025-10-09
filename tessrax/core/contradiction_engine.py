"""
Tessrax Contradiction Engine v5.0
Runtime-Hardened + Self-Auditing Build
Author: Joshua Vetos / Tessrax Governance Agent
License: CC-BY-4.0

Purpose:
  • Detect and emit contradiction events with signed receipts.
  • Sandbox all I/O and quarantine failures safely.
  • Integrate tracer → governance kernel event hooks.
"""

import os, json, time, hashlib, threading, logging
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter, REGISTRY

# Patched Tessrax modules are assumed already present (ILedger, Tracer, etc.)
from tessrax.core.interfaces import ILedger
from tessrax.core.receipts import verify_receipt, NonceRegistry, RevocationRegistry
from tessrax.core.resource_guard import ResourceMonitor, ensure_in_sandbox
from tessrax.utils.tracer import Tracer, trace

# ------------------------------------------------------------
# Logging + Metrics
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ContradictionEngine")

# register metric only once
if not REGISTRY.get_sample_value("tessrax_contradictions_total"):
    CONTRADICTION_EVENTS_PROCESSED = Counter(
        "tessrax_contradictions_total",
        "Number of contradiction events processed by the engine"
    )
else:
    class _Dummy:
        def inc(self): pass
    CONTRADICTION_EVENTS_PROCESSED = _Dummy()

# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------
class ContradictionEngineError(Exception): ...
class QuarantineViolation(Exception): ...

# ------------------------------------------------------------
# Core Engine
# ------------------------------------------------------------
class ContradictionEngine:
    """Hardened contradiction-processing runtime with signed ledger outputs."""

    def __init__(
        self,
        *,
        ledger: ILedger,
        signing_key_hex: str,
        nonce_registry: NonceRegistry,
        revocation_registry: RevocationRegistry,
        ruleset: Optional[List[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]] = None,
        metabolize_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        verify_strict: bool = True,
        name: str = "contradiction_engine",
    ):
        if not all([ledger, signing_key_hex, nonce_registry, revocation_registry]):
            raise ContradictionEngineError("Missing critical dependencies.")

        self.name = name
        self.ledger = ledger
        self.ruleset = ruleset or []
        self.metabolize_fn = metabolize_fn
        self.verify_strict = verify_strict
        self._lock = threading.Lock()
        self._running = True

        self.signing_key = SigningKey(bytes.fromhex(signing_key_hex), encoder=HexEncoder)
        self.verify_key = self.signing_key.verify_key.encode(encoder=HexEncoder).decode()

        sandbox_root = Path(os.environ.get("TESSRAX_SANDBOX_ROOT", "data/sandbox"))
        quarantine_file = os.environ.get("TESSRAX_QUARANTINE_FILENAME", "quarantine.jsonl")
        self._quarantine_path = ensure_in_sandbox(sandbox_root / quarantine_file, sandbox_root)

        self.tracer = Tracer(ledger=self.ledger, private_key_hex=signing_key_hex, executor_id=self.name)
        self._monitor = ResourceMonitor(self.name)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _sha256(obj: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    # --------------------------------------------------------
    # Signing / Verification
    # --------------------------------------------------------
    @trace
    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        serialized = json.dumps(payload, sort_keys=True).encode()
        signature = self.signing_key.sign(serialized).hex()
        payload.update({"signature": signature, "verify_key": self.verify_key})
        return payload

    @trace
    def _verify_event(self, event: Dict[str, Any]) -> bool:
        try:
            return verify_receipt(event.get("receipt"), strict=self.verify_strict)
        except Exception as e:
            logger.error(f"Receipt verification failed: {e}", exc_info=True)
            raise ContradictionEngineError(str(e))

    # --------------------------------------------------------
    # Detection / Emission
    # --------------------------------------------------------
    @trace
    def _detect(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = []
        for rule in self.ruleset:
            try:
                out = rule(event)
                if out:
                    results.append(out)
            except Exception as e:
                logger.warning(f"Rule {rule.__name__} error: {e}", exc_info=True)
        return results

    @trace
    def _emit(self, contradiction: Dict[str, Any]) -> None:
        base = {"timestamp": self._now(), "type": "contradiction", "payload": contradiction}
        signed = self._sign_payload(base)
        try:
            self.ledger.add_event(signed)
            CONTRADICTION_EVENTS_PROCESSED.inc()
        except Exception as e:
            logger.error(f"Ledger add failed: {e}", exc_info=True)
            raise ContradictionEngineError(str(e))
        if self.metabolize_fn:
            try:
                self.metabolize_fn(contradiction)
            except Exception as e:
                logger.warning(f"Metabolism failure: {e}", exc_info=True)

    # --------------------------------------------------------
    # Quarantine
    # --------------------------------------------------------
    @trace
    def _quarantine(self, event: Dict[str, Any], reason: str):
        record = {"timestamp": self._now(), "reason": reason, "event": event}
        try:
            self._quarantine_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.critical(f"Quarantine write failure: {e}", exc_info=True)
            raise QuarantineViolation(str(e))

    # --------------------------------------------------------
    # Main Pipeline
    # --------------------------------------------------------
    def _run_once(self, event: Dict[str, Any]):
        if not self._verify_event(event):
            self._quarantine(event, "Receipt verification failed")
            return
        try:
            for contradiction in self._detect(event):
                self._emit(contradiction)
        except Exception as e:
            self._quarantine(event, f"Pipeline error: {e}")

    def run_batch(self, events: List[Dict[str, Any]]):
        with self._lock:
            for ev in events:
                try:
                    self._run_once(ev)
                except QuarantineViolation:
                    self.stop()
                    break
                except Exception as e:
                    self._quarantine(ev, f"Unexpected batch error: {e}")

    def run_forever(self, source: Callable[[], Dict[str, Any]], delay: float = 1.0):
        logger.info(f"{self.name} live... polling every {delay}s")
        while self._running:
            try:
                ev = source()
                if ev:
                    self.run_batch([ev])
                time.sleep(delay)
            except KeyboardInterrupt:
                logger.info("Interrupted → stopping.")
                self.stop()
            except Exception as e:
                logger.error(f"Runtime loop error: {e}", exc_info=True)
                self._quarantine({"error": str(e)}, "Runtime failure")

    def stop(self):
        logger.info(f"{self.name} shutting down.")
        self._running = False
        if getattr(self, "tracer", None):
            self.tracer.stop()

    # --------------------------------------------------------
    # Verification / Stats
    # --------------------------------------------------------
    def verify_chain(self) -> bool:
        try:
            return self.ledger.verify_chain()
        except Exception as e:
            logger.error(f"Ledger chain verification error: {e}")
            raise ContradictionEngineError(str(e))

    def get_stats(self) -> Dict[str, Any]:
        try:
            all_events = self.ledger.get_all_events(verify=False)
            contradictions = [e for e in all_events if e.get("type") == "contradiction"]
            scars = [e for e in all_events if e.get("type") == "scar"]
            quarantine_size = self._quarantine_path.stat().st_size if self._quarantine_path.exists() else 0
            return {
                "contradictions": len(contradictions),
                "scars": len(scars),
                "quarantine_bytes": quarantine_size,
                "chain_valid": self.ledger.verify_chain(),
            }
        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            raise ContradictionEngineError(str(e))


# ------------------------------------------------------------
# Demo Harness
# ------------------------------------------------------------
if __name__ == "__main__":
    dummy_ledger = ILedger()
    dummy_nonce = NonceRegistry()
    dummy_revoke = RevocationRegistry()
    key = SigningKey.generate().encode(encoder=HexEncoder).decode()

    engine = ContradictionEngine(
        ledger=dummy_ledger,
        signing_key_hex=key,
        nonce_registry=dummy_nonce,
        revocation_registry=dummy_revoke,
        ruleset=[lambda e: {"msg": "Test contradiction"} if "fail" in e else None],
    )

    sample_event = {"receipt": "ok", "payload": {"fail": True}}
    engine.run_batch([sample_event])
    print("🧩 Stats:", json.dumps(engine.get_stats(), indent=2))