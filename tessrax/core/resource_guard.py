# tessrax/core/resource_guard.py
"""
Tessrax Resource Monitor v4.0 — Cryptographically signed resource observability and tamper-evident audit logging.

What this module *actually does*:
 • MONITORS and logs voluntary resource usage (it does NOT enforce OS-level limits)
 • Provides tamper-evident, signed JSONL ledger of activity
 • Optionally verifies chain integrity on each write (verify_on_write=True)
 • Supports cryptographic event signing (Ed25519)
 • Preserves cross-ledger continuity on rotation
 • Thread-safe, fsync-hardened, sandbox-validated
 • Prometheus metrics for observability

Token Semantics:
 ─────────────────────────────────────────────────────────────
 “tokens” = arbitrary, application-defined cost units (e.g., API calls, MB memory)
 “steps”  = number of discrete logical operations or iterations
 Enforcement of actual system limits must be done via OS isolation (e.g., cgroups).
"""

import json
import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.encoding import HexEncoder
from prometheus_client import Counter, Gauge

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

RESOURCE_TOKENS_CONSUMED = Counter("tessrax_resource_tokens_total", "Total resource tokens consumed")
RESOURCE_STEPS_CONSUMED = Counter("tessrax_resource_steps_total", "Total resource steps executed")
RESOURCE_MONITOR_VIOLATIONS = Counter("tessrax_resource_monitor_violations_total", "Total resource limit violations")
AUDIT_EVENTS_TOTAL = Counter("tessrax_audit_events_total", "Total audit events recorded")
LEDGER_LAST_VERIFIED = Gauge("tessrax_ledger_last_verified_timestamp", "Last ledger verification timestamp")

# ---------------------------------------------------------------------
# Structured Errors
# ---------------------------------------------------------------------

class MonitorError(Exception):
    def __init__(self, message: str, **context):
        super().__init__(message)
        self.context = context
    def to_json(self) -> Dict[str, Any]:
        return {"error": self.__class__.__name__, "message": str(self), **self.context}

class ResourceLimitExceeded(MonitorError): ...
class SandboxViolation(MonitorError): ...
class LedgerIntegrityError(MonitorError): ...

# ---------------------------------------------------------------------
# Sandbox Utilities
# ---------------------------------------------------------------------

def ensure_in_sandbox(path: Path, sandbox_root: Path) -> Path:
    """Ensure a path stays confined within sandbox_root."""
    resolved_root = sandbox_root.resolve()
    resolved_target = path.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError:
        RESOURCE_MONITOR_VIOLATIONS.inc()
        raise SandboxViolation(
            f"Path {resolved_target} escaped sandbox {resolved_root}",
            path=str(resolved_target),
            sandbox=str(resolved_root),
        )
    return resolved_target

# ---------------------------------------------------------------------
# Canonical JSON helper
# ---------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

# ---------------------------------------------------------------------
# Tamper-Evident Ledger (Signed + Verifiable)
# ---------------------------------------------------------------------

class AuditLedger:
    """
    Signed, hash-chained append-only ledger.
    Each event includes: prev_hash + Ed25519 signature (optional).
    Supports rotation with continuity and verify_on_write mode.
    """

    def __init__(
        self,
        ledger_path: Path,
        sandbox_root: Optional[Path] = None,
        actor: Optional[str] = None,
        max_lines: int = 100_000,
        verify_on_write: bool = False,
        signing_key_hex: Optional[str] = None
    ):
        self.sandbox_root = sandbox_root or Path("data/sandbox")
        self.path = ensure_in_sandbox(ledger_path, self.sandbox_root)
        os.makedirs(self.sandbox_root, exist_ok=True)
        self.actor = actor or "anonymous"
        self.max_lines = max_lines
        self.verify_on_write = verify_on_write
        self._lock = threading.Lock()

        self.signing_key: Optional[SigningKey] = (
            SigningKey(signing_key_hex, encoder=HexEncoder) if signing_key_hex else None
        )
        self.verify_key: Optional[str] = (
            self.signing_key.verify_key.encode(encoder=HexEncoder).decode()
            if self.signing_key else None
        )

        if not self.path.exists():
            self._initialize_ledger()
        else:
            self.verify_chain()

    # ------------------------------------------------------------
    def write_event(self, action: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Append an event, sign it (if key present), optionally verify chain."""
        with self._lock:
            prev_line = self._read_last_line()
            prev_hash = hashlib.sha256(prev_line.encode("utf-8")).hexdigest() if prev_line else None

            event = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "actor": self.actor,
                "action": action,
                "prev_hash": prev_hash,
                **(payload or {}),
            }

            if self.signing_key:
                signature = self.signing_key.sign(_canonical_json(event).encode()).signature.hex()
                event["signature"] = signature
                event["signer"] = self.verify_key

            line = json.dumps(event, separators=(",", ":")) + "\n"
            self._atomic_append(line)
            AUDIT_EVENTS_TOTAL.inc()

            if self.verify_on_write:
                self.verify_chain()

            if self._line_count() >= self.max_lines:
                self._rotate_ledger()

    # ------------------------------------------------------------
    def verify_chain(self) -> bool:
        """Verify chain integrity and signatures (if present)."""
        prev_hash = None
        with open(self.path, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    raise LedgerIntegrityError("Corrupted JSON line", line_number=i)

                # Chain check
                if event.get("prev_hash") != prev_hash:
                    raise LedgerIntegrityError("Hash chain broken", line_number=i, event=event)

                # Signature check
                if "signature" in event and "signer" in event:
                    vk = VerifyKey(event["signer"].encode(), encoder=HexEncoder)
                    unsigned_event = {k:v for k,v in event.items() if k not in {"signature","signer"}}
                    try:
                        vk.verify(_canonical_json(unsigned_event).encode(), bytes.fromhex(event["signature"]))
                    except Exception as e:
                        raise LedgerIntegrityError("Invalid signature", line_number=i, signer=event.get("signer"), error=str(e))

                prev_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
        LEDGER_LAST_VERIFIED.set_to_current_time()
        return True

    def verify_full_history(self) -> bool:
        """Verify all archived ledgers plus current one."""
        archive_dir = self.path.parent / "archive"
        if archive_dir.exists():
            for arch in sorted(archive_dir.glob(f"{self.path.stem}_*.jsonl")):
                original = self.path
                self.path = arch
                try: self.verify_chain()
                finally: self.path = original
        return self.verify_chain()

    # ------------------------------------------------------------
    def _initialize_ledger(self):
        genesis = {"genesis": True, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "prev_hash": None}
        if self.signing_key:
            sig = self.signing_key.sign(_canonical_json(genesis).encode()).signature.hex()
            genesis["signature"], genesis["signer"] = sig, self.verify_key
        self._atomic_append(json.dumps(genesis) + "\n")

    def _read_last_line(self) -> str:
        if not self.path.exists(): return ""
        with open(self.path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else ""

    def _line_count(self) -> int:
        if not self.path.exists(): return 0
        with open(self.path, "rb") as f:
            return sum(1 for _ in f if b"\n" in _)

    def _rotate_ledger(self):
        """Rotate ledger file safely, linking continuity."""
        last_line = self._read_last_line()
        last_hash = hashlib.sha256(last_line.encode()).hexdigest() if last_line else None

        archive_dir = ensure_in_sandbox(self.path.parent / "archive", self.sandbox_root)
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = ensure_in_sandbox(archive_dir / f"{self.path.stem}_{int(time.time())}.jsonl", self.sandbox_root)
        os.rename(self.path, archive_path)

        genesis = {
            "genesis": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prev_hash": last_hash,
            "prev_ledger": str(archive_path),
        }
        if self.signing_key:
            sig = self.signing_key.sign(_canonical_json(genesis).encode()).signature.hex()
            genesis["signature"], genesis["signer"] = sig, self.verify_key
        self._atomic_append(json.dumps(genesis, separators=(",", ":")) + "\n")
        self.write_event("rotation_complete", {"archived": str(archive_path)})

    def _atomic_append(self, line: str):
        with open(self.path, "ab") as f:
            f.write(line.encode())
            f.flush()
            os.fsync(f.fileno())
        try:
            fd = os.open(str(self.path.parent), os.O_RDONLY)
            os.fsync(fd); os.close(fd)
        except Exception: pass

# ---------------------------------------------------------------------
# Resource Monitor (formerly Guard)
# ---------------------------------------------------------------------

class ResourceMonitor:
    """
    Thread-safe voluntary resource monitor with cryptographically signed audit ledger.
    This does *not* enforce OS-level limits — it provides transparent observability.

    Example:
        with ResourceMonitor(1000, 200) as rm:
            for i in range(50):
                rm.consume(tokens=10, steps=1)
    """

    def __init__(
        self,
        max_tokens: int,
        max_steps: int,
        ledger: Optional[AuditLedger] = None,
        name: Optional[str] = None,
    ):
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.name = name or "anonymous"
        ledger_path = Path(f"data/sandbox/{self.name}_ledger.jsonl")
        self.ledger = ledger or AuditLedger(ledger_path, actor=self.name)
        self._tokens_used = 0
        self._steps_used = 0
        self._lock = threading.Lock()

    def consume(self, tokens: int = 0, steps: int = 1):
        """Track voluntary resource use; raise if over limits."""
        with self._lock:
            self._tokens_used += tokens
            self._steps_used += steps
            RESOURCE_TOKENS_CONSUMED.inc(tokens)
            RESOURCE_STEPS_CONSUMED.inc(steps)
            if self._tokens_used > self.max_tokens or self._steps_used > self.max_steps:
                field = "tokens" if self._tokens_used > self.max_tokens else "steps"
                limit = self.max_tokens if field == "tokens" else self.max_steps
                actual = self._tokens_used if field == "tokens" else self._steps_used
                RESOURCE_MONITOR_VIOLATIONS.inc()
                self.ledger.write_event("violation", {"field": field, "limit": limit, "actual": actual})
                raise ResourceLimitExceeded("Resource limit exceeded", field=field, limit=limit, actual=actual)

    def reset(self):
        with self._lock:
            before = {"tokens": self._tokens_used, "steps": self._steps_used}
            self._tokens_used = self._steps_used = 0
            self.ledger.write_event("reset", {"before": before, "after": {"tokens": 0, "steps": 0}})

    def __enter__(self):
        self.ledger.write_event("start", {"actor": self.name, "limits": {"tokens": self.max_tokens, "steps": self.max_steps}})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ledger.write_event(
            "end",
            {"tokens_used": self._tokens_used, "steps_used": self._steps_used, "error": str(exc_val) if exc_val else None},
        )
        return False