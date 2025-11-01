"""Continuous watchdog monitor that triggers the Tessrax governance audit."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess, run

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard

    class FileSystemEvent:
        def __init__(self, src_path: str, is_directory: bool = False) -> None:
            self.src_path = src_path
            self.is_directory = is_directory

    class FileSystemEventHandler:  # type: ignore[misc]
        def on_modified(self, event: FileSystemEvent) -> None:
            return None

        def on_created(self, event: FileSystemEvent) -> None:
            return None

    class Observer:  # type: ignore[misc]
        def schedule(self, *args, **kwargs) -> None:
            raise RuntimeError("watchdog is required to use the observer")

        def start(self) -> None:
            raise RuntimeError("watchdog is required to use the observer")

        def stop(self) -> None:
            return None

        def join(self) -> None:
            return None

    _WATCHDOG_AVAILABLE = False
else:
    _WATCHDOG_AVAILABLE = True

AUDIT_COMMAND: Sequence[str] = ("python", "tessrax/core/governance_controller.py")


class PythonFileEventHandler(FileSystemEventHandler):
    """React to Python file events by running the governance audit."""

    def __init__(
        self,
        command: Sequence[str],
        runner: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        super().__init__()
        self._command = list(command)
        self._runner = runner or _run_governance_audit

    def _handle_path(self, path: str) -> bool:
        if not path.endswith(".py"):
            return False
        print("[Tessrax] Change detected — re-auditing...")
        success = self._runner(self._command)
        if success:
            print("✅ Governance audit passed")
        else:
            print("❌ Governance audit failed")
        return True

    def on_modified(
        self, event: FileSystemEvent
    ) -> None:  # pragma: no cover - integration behaviour
        if not event.is_directory:
            self._handle_path(event.src_path)

    def on_created(
        self, event: FileSystemEvent
    ) -> None:  # pragma: no cover - integration behaviour
        if not event.is_directory:
            self._handle_path(event.src_path)


def _run_governance_audit(command: Sequence[str]) -> bool:
    try:
        result: CompletedProcess[bytes] = run(command, check=False)
    except (OSError, CalledProcessError) as exc:
        print(f"[Tessrax] Failed to execute governance audit: {exc}", file=sys.stderr)
        return False
    return result.returncode == 0


def run_watchdog(
    directory: str | os.PathLike[str] = ".", *, command: Sequence[str] = AUDIT_COMMAND
) -> None:
    if not _WATCHDOG_AVAILABLE:
        raise SystemExit(
            "[Tessrax] watchdog dependency is required. Install it via 'pip install watchdog'."
        )
    observer = Observer()
    handler = PythonFileEventHandler(command)
    observer.schedule(handler, str(directory), recursive=True)
    observer.start()
    print("[Tessrax] Watchdog active — monitoring Python files...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        observer.join()


def _self_test() -> bool:
    triggered: dict[str, bool] = {"called": False}

    def fake_runner(_: Sequence[str]) -> bool:
        triggered["called"] = True
        return True

    handler = PythonFileEventHandler(("python", "-V"), runner=fake_runner)
    handled_python = handler._handle_path(str(Path("example.py")))
    assert handled_python is True, "Handler should act on .py files"
    assert triggered["called"] is True, "Fake runner should be invoked for .py files"
    handled_text = handler._handle_path(str(Path("notes.txt")))
    assert handled_text is False, "Handler should ignore non-Python files"
    return True


if __name__ == "__main__":
    run_watchdog()
