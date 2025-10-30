# Plugin Sandbox Security Guarantees

Tessrax executes governance plugins inside a RestrictedPython sandbox with the
following constraints:

- **Memory limit:** 100 MB address-space cap enforced via `RLIMIT_AS`.
- **CPU limit:** 30 seconds of CPU time via `RLIMIT_CPU`.
- **Filesystem:** plugins may only read/write inside `/tmp/plugin_*` paths.
- **Network:** module imports are restricted to a safe allow-list (`math`,
  `statistics`), preventing socket or HTTP usage.
- **Side effects:** plugins must communicate results by assigning to a `result`
  variable; no globals leak back to the runtime.

The `tests/test_plugin_sandbox.py` suite validates these guarantees and is wired
into CI so every pull request executes the sandboxed plugin compatibility suite
prior to merge.
