"""
tessrax/core/resource_guard.py
------------------------------
Tessrax Resource Guard (v2.0)
Protects system stability by enforcing CPU, memory, and file safety limits.

✓ Safe sandbox path enforcement
✓ Context manager for resource ceilings
✓ Optional psutil integration for live monitoring
✓ Structured exceptions for forensic logging
"""

import os
import time
import json
import signal
import resource
import threading
from pathlib import Path
from typing import Optional, Dict, Any


# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------
class ResourceLimitExceeded(Exception):
    """Raised when CPU or memory limits are exceeded."""
    pass


class SandboxViolation(Exception):
    """Raised when a file path escapes the allowed sandbox root."""
    pass


# ------------------------------------------------------------
# Path Enforcement
# ------------------------------------------------------------
def ensure_in_sandbox(path: Path, sandbox_root: Path) -> Path:
    """
    Validates that a path is inside the sandbox root directory.
    Creates directories if missing.
    """
    sandbox_root = sandbox_root.resolve()
    candidate = path.resolve()

    if not str(candidate).startswith(str(sandbox_root)):
        raise SandboxViolation(f"Path {candidate} escapes sandbox {sandbox_root}")

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


# ------------------------------------------------------------
# Resource Monitor
# ------------------------------------------------------------
class ResourceMonitor:
    """
    Monitors CPU and memory usage, enforces ceilings, and exposes
    a structured snapshot for telemetry or receipts.
    """

    def __init__(self, cpu_limit: float = 0.9, mem_limit_mb: int = 512, poll_interval: float = 0.25):
        """
        cpu_limit: fraction of a core (0.0–1.0)
        mem_limit_mb: max allowed resident set size (MB)
        """
        self.cpu_limit = cpu_limit
        self.mem_limit_mb = mem_limit_mb
        self.poll_interval = poll_interval
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._violation: Optional[str] = None

    # --------------------------------------------------------
    # Internal monitoring logic
    # --------------------------------------------------------
    def _monitor(self):
        import psutil  # optional but preferred
        process = psutil.Process(os.getpid())

        while self._active:
            cpu = process.cpu_percent(interval=None) / 100.0
            mem = process.memory_info().rss / (1024 ** 2)
            if cpu > self.cpu_limit:
                self._violation = f"CPU limit exceeded: {cpu:.2f} > {self.cpu_limit}"
                self._active = False
                break
            if mem > self.mem_limit_mb:
                self._violation = f"Memory limit exceeded: {mem:.1f}MB > {self.mem_limit_mb}MB"
                self._active = False
                break
            time.sleep(self.poll_interval)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def start(self):
        """Begin background monitoring in a daemon thread."""
        try:
            import psutil
        except ImportError:
            # Fallback: no psutil → only static rlimit guard
            self._apply_rlimits()
            return
        if self._active:
            return
        self._active = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring thread, raise if violation occurred."""
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._violation:
            raise ResourceLimitExceeded(self._violation)

    def snapshot(self) -> Dict[str, Any]:
        """Return a structured resource snapshot."""
        import psutil
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss / (1024 ** 2)
        cpu = p.cpu_percent(interval=0.05) / 100.0
        return {"cpu": cpu, "memory_mb": mem, "timestamp": time.time()}

    # --------------------------------------------------------
    # POSIX Fallback
    # --------------------------------------------------------
    def _apply_rlimits(self):
        """Fallback limit enforcement using resource module."""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            mem_bytes = self.mem_limit_mb * 1024 ** 2
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
        except Exception:
            pass  # not supported on all systems

    # --------------------------------------------------------
    # Context Manager Support
    # --------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    sandbox_root = Path("data/sandbox")
    test_path = ensure_in_sandbox(sandbox_root / "demo" / "file.txt", sandbox_root)
    print(f"Validated sandbox path: {test_path}")

    monitor = ResourceMonitor(cpu_limit=0.8, mem_limit_mb=256)
    with monitor:
        data = [x for x in range(1_000_000)]
        print("Snapshot:", json.dumps(monitor.snapshot(), indent=2))
    print("Completed without violation.")