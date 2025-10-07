"""
tessrax/utils/tracer.py
-----------------------
Runtime tracer for Tessrax Engine.

Provides:
- Tracer class for asynchronous trace logging
- trace decorator for lightweight instrumentation
"""

import json
import time
import threading
from typing import Callable, Any, Dict


class Tracer:
    """Minimal asynchronous tracer for runtime event logging."""

    def __init__(self, enable_async: bool = True, ledger=None, private_key_hex=None, executor_id=None):
        self.enable_async = enable_async
        self._queue = []
        self._lock = threading.Lock()
        self._active = True
        self.ledger = ledger
        self.private_key_hex = private_key_hex
        self.executor_id = executor_id

        if self.enable_async:
            self._thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._thread.start()

    def record(self, event_type: str, payload: Dict[str, Any]):
        """Record a trace event."""
        entry = {
            "timestamp": time.time(),
            "type": event_type,
            "payload": payload,
        }
        with self._lock:
            self._queue.append(entry)

    def _flush_loop(self):
        """Asynchronous flushing loop."""
        while self._active:
            time.sleep(0.2)
            self.flush()

    def flush(self):
        """Print and clear all queued trace events."""
        with self._lock:
            while self._queue:
                event = self._queue.pop(0)
                print(f"[TRACE] {json.dumps(event)}")

    def stop(self):
        """Stop background thread and flush remaining events."""
        self._active = False
        if self.enable_async and hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)
        self.flush()


# ============================================================
# trace Decorator
# ============================================================
def trace(func: Callable) -> Callable:
    """Simple decorator for tracing function calls."""
    def wrapper(*args, **kwargs):
        print(f"[TRACE_DECORATOR] Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
