from __future__ import annotations

import builtins
import os
import shutil
import tempfile
import traceback
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any

# resource is POSIX-only; guard for portability
try:  # pragma: no cover - imported at module load
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore[assignment]

from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins
from RestrictedPython import utility_builtins
from RestrictedPython.Eval import default_guarded_getattr
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import guarded_setattr

MEMORY_LIMIT_BYTES = 100 * 1024 * 1024
CPU_LIMIT_SECONDS = 30

_ALLOWED_MODULES = MappingProxyType({"math": __import__("math"), "statistics": __import__("statistics")})


@contextmanager
def _resource_limits(memory_bytes: int, cpu_seconds: int):
    if resource is None:
        yield
        return

    old_limits = {
        resource.RLIMIT_AS: resource.getrlimit(resource.RLIMIT_AS),
        resource.RLIMIT_CPU: resource.getrlimit(resource.RLIMIT_CPU),
    }
    try:
        as_soft, as_hard = old_limits[resource.RLIMIT_AS]
        cpu_soft, cpu_hard = old_limits[resource.RLIMIT_CPU]
        new_as_soft = min(memory_bytes, as_hard if as_hard != resource.RLIM_INFINITY else memory_bytes)
        new_cpu_soft = min(cpu_seconds, cpu_hard if cpu_hard != resource.RLIM_INFINITY else cpu_seconds)
        resource.setrlimit(resource.RLIMIT_AS, (new_as_soft, as_hard))
        resource.setrlimit(resource.RLIMIT_CPU, (new_cpu_soft, cpu_hard))
        yield
    finally:
        for key, value in old_limits.items():
            resource.setrlimit(key, value)


def _restricted_import(name: str, globals: dict[str, Any] | None = None, locals: dict[str, Any] | None = None, fromlist: tuple[str, ...] = (), level: int = 0):  # noqa: D401 - compatibility shim
    if name in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[name]
    raise ImportError(f"Module '{name}' is not permitted in the Tessrax sandbox")


def _sandbox_builtins() -> dict[str, Any]:
    return {
        "__builtins__": {
            **safe_builtins,
            **utility_builtins,
            "__import__": _restricted_import,
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
        },
        "_getattr_": default_guarded_getattr,
        "_getitem_": default_guarded_getitem,
        "_setattr_": guarded_setattr,
    }


def _sandbox_open(tmp_root: Path):
    def _open(path: str, mode: str = "r", *args: Any, **kwargs: Any):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (tmp_root / candidate).resolve()
        if not str(candidate).startswith(str(tmp_root)):
            raise PermissionError("Plugins may only write within /tmp/plugin_* directories")
        os.makedirs(candidate.parent, exist_ok=True)
        return builtins.open(candidate, mode, *args, **kwargs)

    return _open


def _run_in_child(code: str, payload: dict[str, Any], conn) -> None:
    try:
        bytecode = compile_restricted(code, filename="<plugin>", mode="exec")
        tmp_root = Path(tempfile.mkdtemp(prefix="plugin_", dir="/tmp"))
        globals_dict = _sandbox_builtins()
        globals_dict.update({"payload": payload, "__name__": "__plugin__"})
        globals_dict["__builtins__"]["open"] = _sandbox_open(tmp_root)

        with _resource_limits(MEMORY_LIMIT_BYTES, CPU_LIMIT_SECONDS):
            exec(bytecode, globals_dict)  # noqa: S102 - RestrictedPython safe exec

        if "result" not in globals_dict:
            raise RuntimeError("Plugin must set a 'result' variable")

        conn.send(("ok", globals_dict["result"]))
    except Exception as exc:  # pragma: no cover - marshalled back to parent
        conn.send(("error", (exc, traceback.format_exc())))
    finally:
        conn.close()
        if 'tmp_root' in locals():
            shutil.rmtree(tmp_root, ignore_errors=True)


def execute_plugin(code: str, *, payload: dict[str, Any] | None = None) -> Any:
    """Execute plugin code within a RestrictedPython sandbox."""

    payload = payload or {}
    ctx = get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_run_in_child, args=(code, payload, child_conn))
    process.start()
    child_conn.close()
    process.join(CPU_LIMIT_SECONDS + 5)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Plugin execution exceeded CPU limit")
    if not parent_conn.poll():
        raise RuntimeError("Sandbox did not return a result")
    status, message = parent_conn.recv()
    if status == "error":
        exc, tb = message
        if isinstance(exc, BaseException):
            raise exc
        raise RuntimeError(f"Plugin execution failed:\n{tb}")
    return message
