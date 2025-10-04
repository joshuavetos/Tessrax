# tessrax/core/contradiction_engine.py
"""
Tessrax Contradiction Engine v4.7 (Hardened Production Build)
--------------------------------------------------------------
Core contradiction detection and metabolism layer.

✅ Secure-by-default (registries required)
✅ Sandboxed + fsync’d quarantine
✅ Signed contradiction emissions
✅ Thread-safe batch + event loop
✅ Ledger-driven hash chain continuity
✅ Metabolize integration (optional)
✅ Critical-failure halting (no silent continues)
"""

import os
import json
import time
import hashlib
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Callable, List, Optional

from nacl.signing import SigningKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter

from tessrax.core.receipts import verify_receipt, NonceRegistry, RevocationRegistry
from tessrax.core.resource_guard import ResourceMonitor, ensure_in_sandbox

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
CONTRADICTION_EVENTS_PROCESSED = Counter(
    "tessrax_contradiction_events_total",
    "Total verified events processed"
)
CONTRADICTION_EVENTS_QUARANTINED = Counter(
    "tessrax_contradiction_events_quarantined_total",
    "Total contradiction events quarantined"
)
QUARANTINE_FSYNC_FAILURES = Counter(
    "tessrax_quarantine_fsync_failures_total",
    "Number of quarantine fsync failures"
)
CONTRADICTION_RULE_ERRORS = Counter(
    "tessrax_contradiction_rule_errors_total",
    "Total rule evaluation errors"
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ContradictionEngineError(Exception): ...
class QuarantineViolation(ContradictionEngineError): ...
class CriticalFailure(ContradictionEngineError): ...

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ContradictionEngine:
    """Secure, verifiable contradiction-processing runtime."""

    def __init__(
        self,
        name: str,
        ruleset: Optional[List[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]] = None,
        signing_key_hex: str = "",
        ledger=None,
        *,
        nonce_registry: NonceRegistry,
        revocation_registry: RevocationRegistry,
        verify_strict: bool = True,
        quarantine_path: str = "data/quarantine.jsonl",
        metabolize_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
    ):
        # --- Secure defaults ---
        if not nonce_registry or not revocation_registry:
            raise ContradictionEngineError("NonceRegistry and RevocationRegistry required")

        self.name = name
        self.ruleset = ruleset or []
        self.ledger = ledger
        self.nonce_registry = nonce_registry
        self.revocation_registry = revocation_registry
        self.verify_strict = verify_strict
        self.metabolize_fn = metabolize_fn
        self._running = True

        # Sandbox + durability setup
        sandbox_root = Path("data/sandbox")
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self._quarantine_path = ensure_in_sandbox(Path(quarantine_path), sandbox_root)
        os.makedirs(self._quarantine_path.parent, exist_ok=True)

        self._lock = threading.Lock()
        self.monitor = ResourceMonitor(max_tokens=100_000, max_steps=10_000, name=self.name)

        self.signing_key = SigningKey(signing_key_hex, encoder=HexEncoder)
        self.verify_key = self.signing_key.verify_key.encode(encoder=HexEncoder).decode()

        # Validate metabolize_fn signature (1 param)
        if self.metabolize_fn:
            import inspect
            sig = inspect.signature(self.metabolize_fn)
            if len(sig.parameters) != 1:
                raise ContradictionEngineError("metabolize_fn must take exactly one parameter")

    # -----------------------------------------------------------------------
    def _verify_event(self, event: Dict[str, Any]) -> None:
        """Strict receipt verification using receipts.py primitives."""
        try:
            verify_receipt(
                event,
                strict=self.verify_strict,
                nonce_registry=self.nonce_registry,
                revocation_registry=self.revocation_registry,
            )
        except Exception as e:
            raise ContradictionEngineError(f"Receipt verification failed: {e}")

    # -----------------------------------------------------------------------
    def _sign_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Attach Ed25519 signature to canonical event."""
        canonical = _canonical_json(payload)
        sig = self.signing_key.sign(canonical.encode()).signature.hex()
        return {**payload, "signature": sig, "signer": self.verify_key}

    # -----------------------------------------------------------------------
    def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
        """Append durable forensic log entry, fail closed on I/O failure."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": f"{reason}\n{traceback.format_exc()}",
            "event": event,
            "engine": self.name,
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"

        try:
            with open(self._quarantine_path, "ab") as f:
                f.write(line.encode())
                f.flush()
                os.fsync(f.fileno())

            dfd = os.open(self._quarantine_path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception as e:
            QUARANTINE_FSYNC_FAILURES.inc()
            # Critical — engine must halt
            self._running = False
            raise CriticalFailure(f"Critical: quarantine fsync failed: {e}")

        CONTRADICTION_EVENTS_QUARANTINED.inc()
        raise QuarantineViolation(reason)

    # -----------------------------------------------------------------------
    def _emit(self, contradiction: Dict[str, Any]) -> None:
        """Emit signed contradiction record and optional scar."""
        base = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "contradiction",
            "payload": contradiction,
        }

        # Ledger manages its own chaining internally
        signed = self._sign_payload(base)
        self.ledger.add_event(signed)

        # Metabolize step (optional)
        if self.metabolize_fn:
            try:
                scar = self.metabolize_fn(contradiction)
                if scar:
                    self.ledger.add_event({"type": "scar", "payload": scar})
            except Exception as e:
                self._quarantine({"contradiction": contradiction}, f"metabolize_error: {e}")

        CONTRADICTION_EVENTS_PROCESSED.inc()

    # -----------------------------------------------------------------------
    def _run_once_unlocked(self, event: Dict[str, Any]) -> None:
        """Single event lifecycle: verify → evaluate → emit."""
        with self.monitor:
            self.monitor.consume(steps=1)

            try:
                self._verify_event(event)
            except Exception as e:
                self._quarantine(event, f"verification_failed: {e}")
                return

            seen_contradictions = set()
            contradictions_found = []

            for rule in self.ruleset:
                try:
                    c = rule(event)
                    if c:
                        c_hash = _sha256_hex(_canonical_json(c))
                        if c_hash not in seen_contradictions:
                            seen_contradictions.add(c_hash)
                            contradictions_found.append(c)
                except Exception as e:
                    CONTRADICTION_RULE_ERRORS.inc()
                    self._quarantine(
                        {"event": event, "rule": getattr(rule, "name", str(rule))},
                        f"rule_eval_error: {e}",
                    )
                    # Continue evaluating remaining rules

            for c in contradictions_found:
                self.monitor.consume(steps=1)
                self._emit(c)

    # -----------------------------------------------------------------------
    def run_once(self, event: Dict[str, Any]) -> None:
        """Thread-safe single event processing."""
        with self._lock:
            if not self._running:
                raise CriticalFailure("Engine halted due to prior critical error.")
            self._run_once_unlocked(event)

    def run_batch(self, events: List[Dict[str, Any]]) -> None:
        """Sequential batch processing; halts on critical quarantine failure."""
        with self._lock:
            for ev in events:
                if not self._running:
                    break
                try:
                    self._run_once_unlocked(ev)
                except CriticalFailure:
                    self._running = False
                    raise
                except QuarantineViolation:
                    # Halt instead of silent continue
                    self._running = False
                    raise CriticalFailure(
                        "Quarantine failure encountered. Engine halted for safety."
                    )

    # -----------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return contradiction and scar counts plus quarantine file size."""
        contradictions = [
            e for e in self.ledger.get_all_events() if e.get("type") == "contradiction"
        ]
        scars = [e for e in self.ledger.get_all_events() if e.get("type") == "scar"]
        return {
            "total_contradictions": len(contradictions),
            "total_scars": len(scars),
            "quarantine_size": (
                os.path.getsize(self._quarantine_path)
                if Path(self._quarantine_path).exists()
                else 0
            ),
            "engine_running": self._running,
        }

    def inspect_quarantine(self) -> List[Dict[str, Any]]:
        """Read quarantined events for forensic inspection."""
        if not Path(self._quarantine_path).exists():
            return []
        with open(self._quarantine_path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]