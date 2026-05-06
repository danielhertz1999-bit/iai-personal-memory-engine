"""Plan 07.1-04 R5 acceptance — compile-output regression trap.

This is the regression-trap that catches a future revert of Phase 7.1's
no-spawn architecture. If `child_process.spawn` reappears in
`mcp-wrapper/dist/bridge.js`, this test FAILS — alerting the developer
(or a future Claude) that someone has reintroduced the TOCTOU spawn
race that explicitly removed.

# Why a compile-output trap, not just a source-level grep?

A source-level grep would also catch the regression, but it would NOT
catch:
  - A spawn call introduced via a transitive import (e.g., a helper
    module that imports `node:child_process` and re-exports a spawn
    wrapper).
  - A spawn call introduced via dynamic `require("child_process")` at
    runtime (which tsc compiles into the JS but a source grep for
    `import { spawn }` would miss).
  - A spawn introduced into a NEW module that bridge.ts imports.

The compiled `dist/bridge.js` is what actually ships and runs. Greping
THAT is the load-bearing assertion.

# Reference

- Plan 07.1-04 Task 3
- 07.1-CONTEXT.md D7.1-07 (bridge.ts spawn-removal scope)
- The mirror source-level assertion lives in Task 1
  (``grep -c 'child_process[.]spawn|^import.*spawn|spawnDaemon'
  mcp-wrapper/src/bridge.ts`` returns 0)
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "mcp-wrapper"

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="bash + npm tooling assumed POSIX (mcp-wrapper build path)",
)


# ---------------------------------------------------------------------------
# Fixture: build the wrapper once per module so all 3 tests reuse the same
# dist/bridge.js artifact. Mirrors the pattern in
# tests/test_socket_subagent_reuse.py:built_wrapper.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_bridge_js() -> Path:
    """Build the TS wrapper once; return the path to compiled bridge.js."""
    if not (WRAPPER / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=WRAPPER, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WRAPPER, check=True)
    dist = WRAPPER / "dist" / "bridge.js"
    assert dist.exists(), (
        "npm run build should have produced dist/bridge.js — actual: "
        f"{list((WRAPPER / 'dist').glob('*.js')) if (WRAPPER / 'dist').exists() else 'no dist dir'}"
    )
    return dist


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_dist_bridge_js_has_no_child_process_spawn(built_bridge_js):
    """REGRESSION TRAP: assert the compiled bridge.js contains zero
    references to child_process.spawn in any of its post-tsc forms.

    Catches:
      - `import { spawn } from "node:child_process"` (ESM, what
        TypeScript writes; tsc with module=ESNext keeps the import)
      - `from "node:child_process"` (any other named import from the
        same module)
      - `require("node:child_process")` (CJS form if module target
        ever changes to CommonJS)
      - `require("child_process")` (legacy CJS form)
      - `child_process.spawn` (after a `.spawn` access on a module
        namespace import)

    All five forms are checked because tsc's exact output bytes depend
    on tsconfig (module=ESNext vs CommonJS), and a future config
    change must NOT silently allow spawn back in.
    """
    text = built_bridge_js.read_text(encoding="utf-8")

    forbidden_substrings = [
        'child_process.spawn',
        'from "node:child_process"',
        "from 'node:child_process'",
        'require("node:child_process")',
        "require('node:child_process')",
        'require("child_process")',
        "require('child_process')",
    ]

    found = [s for s in forbidden_substrings if s in text]
    assert not found, (
        "REGRESSION: dist/bridge.js contains spawn-related substring(s) "
        f"that explicitly removed: {found}. "
        "Someone has re-introduced the TOCTOU spawn race that Phase 7.1's "
        "pure-connector refactor eliminated. Re-read 07.1-CONTEXT.md "
        "D7.1-07 (bridge.ts spawn-removal scope) before pushing."
    )


def test_dist_bridge_js_has_DaemonUnreachableError(built_bridge_js):
    """Assert the compiled bridge.js still contains the
    DaemonUnreachableError class — proves the no-spawn error-throwing
    path is preserved post-build.

    If start() somehow stops throwing (e.g., a future refactor
    silently swallows the connect failure and degrades to a no-op),
    the symptom would be: wrappers boot fine even with no daemon, but
    every tools/call returns daemon_unreachable. That's a regression
    we want to catch at compile-output level.

    The presence of `DaemonUnreachableError` as a string in dist/bridge.js
    verifies the class definition + at least one throw-site survived
    compilation.
    """
    text = built_bridge_js.read_text(encoding="utf-8")

    # Plan 07.1-04 done criteria for Task 1: DaemonUnreachableError
    # appears ≥2 times in the source (class def + at least one throw).
    # Same expectation for the compiled output — tsc preserves named
    # class identifiers exactly.
    count = text.count("DaemonUnreachableError")
    assert count >= 2, (
        f"REGRESSION: dist/bridge.js contains DaemonUnreachableError "
        f"only {count} times (expected >=2: class definition + at least "
        f"one throw-site). The fail-loud error path may have been "
        f"removed or renamed."
    )


def test_dist_bridge_js_has_5000_socket_timeout(built_bridge_js):
    """Assert the SOCKET_CONNECT_TIMEOUT_MS constant is set to 5000ms
    (raised from 250ms in pre-7.1 to cover launchd socket-activation
    cold-start window).

    Anchored to the named constant (`SOCKET_CONNECT_TIMEOUT_MS = 5000`)
    rather than a bare `5000` substring — tsc default does NOT minify
    so the constant declaration survives compilation verbatim, and a
    bare `5000` could match unrelated literals (timestamps, byte
    counts) the compiler emits.

    If this test fails:
      - The constant was renamed: update the assertion AND verify the
        new name is the connect timeout (not idle-shutdown / heartbeat).
      - The value was lowered (e.g., back to 250): re-read CONTEXT.md
        D7.1-07 — 5s is required because launchd cold-spawn of the
        daemon (bge-small embedder load + LanceDB open) is empirically
        3-10s on macOS. A lower timeout will spuriously throw
        DaemonUnreachableError on legitimate cold-starts.
    """
    text = built_bridge_js.read_text(encoding="utf-8")

    # Anchored to the named constant — survives tsc default (no
    # minification, target ES2022).
    assert "SOCKET_CONNECT_TIMEOUT_MS = 5000" in text, (
        "REGRESSION: dist/bridge.js does not contain "
        "'SOCKET_CONNECT_TIMEOUT_MS = 5000'. Either the constant was "
        "renamed, the value was changed, or tsc minification was "
        "enabled (which would also break the source-level grep done "
        "criteria in Task 1). requires 5000ms to cover "
        "launchd socket-activation cold-start window — see "
        "07.1-CONTEXT.md D7.1-07."
    )
