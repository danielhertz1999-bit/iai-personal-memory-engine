"""Regression fence — no bare sync calls to known-blocking functions
inside `async def` in daemon-side modules.

Mechanism: parse target Python files with ast.parse, walk AsyncFunctionDef
nodes, check every Call node against BLOCKING_NAMES, exempt calls inside
`await asyncio.to_thread(...)` argument position. Fail the test on any
unallowed bare-sync.

Allowlist sites must carry measurement evidence. Format: (file, async_fn, callee).

Background: prevents the daemon-CPU-saturation regression. The smoking-gun
call was `retrieve.build_runtime_graph(store)` inside `_hippea_cascade_loop`
(an asyncio task) — it is now wrapped in
`await asyncio.to_thread(retrieve.build_runtime_graph, store)`. This fence
catches re-introduction.

BLOCKING_NAMES ships populated, not empty, so the fence has teeth. Adding
further entries requires both (a) classification as `wrapped` and
(b) measurement evidence > 50 ms.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"

# Modules transitively reachable from the daemon asyncio tasks.
# Update this tuple when a daemon task touches a new module.
DAEMON_REACHABLE: tuple[str, ...] = (
    "daemon.py",
    "dream.py",
    "identity_audit.py",
    "hippea_cascade.py",
    "socket_server.py",
    "concurrency.py",
    "insight.py",
    # maintenance.py is the home of optimize_lance_storage, a sync helper
    # that does 30+ s of storage file I/O. The fence walks files
    # transitively reachable from daemon tasks; daemon.main() and
    # identity_audit.continuous_audit both invoke optimize_lance_storage, so
    # the helper itself is in scope. The helper is sync def (correctly --
    # callers wrap in asyncio.to_thread); listing it here keeps any accidental
    # async-side helper inside maintenance.py covered too.
    "maintenance.py",
)

# Functions that are SYNCHRONOUS AND HEAVY. Every Call to one of these
# inside `async def` MUST be inside `await asyncio.to_thread(...)`.
#
# Primary entry: build_runtime_graph (the smoking gun — an 8-13 s graph
# traversal that drove the CPU saturation). Further entries are added
# as-measured. Each entry requires:
# - classification as `wrapped`
# - empirical measurement > 50 ms in the worst case
# - confirmation it is a SYNC `def` (not `async def` — those are
# enforced by mypy/pyright type checks, not this fence)
BLOCKING_NAMES: frozenset[str] = frozenset({
    "build_runtime_graph",
    # optimize_lance_storage runs a storage-compaction pass that can take
    # 30+ s. Both production call sites (daemon.main() startup,
    # identity_audit.continuous_audit periodic body) wrap this helper in
    # `await asyncio.to_thread(...)`; the fence catches future re-introduction
    # of a bare sync call.
    "optimize_lance_storage",
    # daemon-state-io fence: load_state reads STATE_PATH.read_text()
    # sync; save_state does tempfile + json.dump + fsync + os.replace sync.
    # On macOS APFS each fsync is 5-50 ms typical, can spike to 200-500 ms
    # under load — async coroutines doing inline sync file I/O on the event
    # loop block the main thread. Sites across daemon.py, concurrency.py,
    # insight.py, lifecycle.py are each wrapped in
    # `await asyncio.to_thread(load_state)` /
    # `await asyncio.to_thread(save_state, state)`. The state dict is
    # small (~50 keys / ~11 KB) so json.dump's iteration completes well
    # within a single GIL grant in practice; concurrent dispatcher
    # writes land safely without an explicit lock. User-facing impact:
    # status RPC and MCP tool replies stop stalling on event-loop file I/O.
    "load_state",
    "save_state",
})

# Allowlisted bare-sync sites with measurement evidence.
# Format: (file_basename, function_name, callee_name).
#
# Currently empty: the verified `safe-fast` sites are all for `write_event`,
# which is not in BLOCKING_NAMES (so the fence never sees them). The
# `sigma.compute_and_emit` chain is sync def, also not seen by the fence.
#
# A future ALLOWLIST entry would be required ONLY when a callee in
# BLOCKING_NAMES has a verified `safe-fast` site (< 50 ms measured)
# inside an `async def` body.
ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset()


def _callable_name(func: ast.expr) -> str | None:
    """Extract the terminal callable name from a Call.func node.

    Handles `foo()`, `mod.foo()`, and `mod.sub.foo()` — returning
    the terminal name in the chain. Returns None for complex
    expressions (lambdas, subscripts) that won't match the blocking
    list anyway.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class BareBlockingCallFinder(ast.NodeVisitor):
    """Walk the AST and record any blocking-named Call nodes that
    are NOT inside the args of `await asyncio.to_thread(...)`."""

    def __init__(self, file_basename: str) -> None:
        self.file = file_basename
        self.violations: list[tuple[str, str, str, int]] = []
        self._in_async_depth = 0
        self._async_fn_stack: list[str] = []
        self._in_to_thread_args = False

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._in_async_depth += 1
        self._async_fn_stack.append(node.name)
        self.generic_visit(node)
        self._async_fn_stack.pop()
        self._in_async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested sync `def` inside an `async def` is NOT bound by
        # the rule — the sync inner function runs in whatever thread
        # the caller picks.
        prev = self._in_async_depth
        self._in_async_depth = 0
        self.generic_visit(node)
        self._in_async_depth = prev

    def visit_Await(self, node: ast.Await) -> None:
        # Detect `await asyncio.to_thread(fn, args, kwargs)` and
        # exempt the args list from the blocking check.
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "to_thread"
        ):
            prev = self._in_to_thread_args
            self._in_to_thread_args = True
            for arg in node.value.args:
                self.visit(arg)
            for kw in node.value.keywords:
                self.visit(kw)
            self._in_to_thread_args = prev
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _callable_name(node.func)
        current_fn = (
            self._async_fn_stack[-1] if self._async_fn_stack else "?"
        )
        if (
            self._in_async_depth > 0
            and not self._in_to_thread_args
            and name is not None
            and name in BLOCKING_NAMES
            and (self.file, current_fn, name) not in ALLOWLIST
        ):
            self.violations.append((
                self.file,
                current_fn,
                name,
                node.lineno,
            ))
        self.generic_visit(node)


def test_no_bare_blocking_call_in_async_def() -> None:
    """Regression fence: no bare sync calls to BLOCKING_NAMES
    inside `async def` in daemon-side modules."""
    all_violations: list[tuple[str, str, str, int]] = []
    for filename in DAEMON_REACHABLE:
        path = SRC / filename
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=filename)
        finder = BareBlockingCallFinder(filename)
        finder.visit(tree)
        all_violations.extend(finder.violations)
    assert not all_violations, (
        "Regression fence: bare sync calls to blocking functions "
        "inside `async def`. Each violation must be either (a) wrapped "
        "in `await asyncio.to_thread(...)` or (b) added to ALLOWLIST "
        "and classified as `safe-fast` with measurement evidence. "
        "Violations:\n"
        + "\n".join(
            f"  {f}:{ln} async def {fn} -> {callee}()"
            for f, fn, callee, ln in all_violations
        )
    )


def test_blocking_names_set_is_non_empty() -> None:
    """Ship-discipline check: the fence must have teeth.

    Don't ship the fence with empty BLOCKING_NAMES — that's a fence with no
    enforcement surface. The minimum is the smoking gun.
    """
    assert "build_runtime_graph" in BLOCKING_NAMES, (
        "BLOCKING_NAMES must contain 'build_runtime_graph' (the "
        "smoking-gun call that drove the CPU saturation). Fence is useless "
        "without this; the site is wrapped and the fence catches future "
        "re-introduction."
    )


def test_ast_walker_correctly_identifies_to_thread_exemption() -> None:
    """Self-check: confirm the visitor correctly distinguishes
    wrapped from bare-sync calls inside the same async function body.
    """
    snippet = '''
import asyncio
from iai_mcp import retrieve

async def good_path(store):
    # Wrapped — fence MUST exempt this.
    g, a, r = await asyncio.to_thread(retrieve.build_runtime_graph, store)
    return g

async def bad_path(store):
    # Bare-sync — fence MUST flag this.
    g, a, r = retrieve.build_runtime_graph(store)
    return g
'''
    tree = ast.parse(snippet)
    finder = BareBlockingCallFinder("synthetic_snippet.py")
    finder.visit(tree)
    # Only the bad_path call should be flagged.
    violations = list(finder.violations)
    assert len(violations) == 1, (
        f"Expected exactly 1 violation (in bad_path); got "
        f"{len(violations)}: {violations}"
    )
    f, fn, callee, ln = violations[0]
    assert fn == "bad_path", f"Expected violation in bad_path; got {fn}"
    assert callee == "build_runtime_graph", (
        f"Expected violation on build_runtime_graph; got {callee}"
    )
