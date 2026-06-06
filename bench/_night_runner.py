"""Tonight-only bench wrapper that forces buffer flush before queries.

Background:
   (Hippo storage migration) made MemoryStore.insert buffered.
  Inserts accumulate in an in-process buffer that flushes on size threshold
  (IAI_MCP_RECORD_BUFFER_MAX, default 500), time threshold (5 s), or
  explicit flush_record_buffer(store). Existing benches do
  "insert N records then query" with N below the size threshold and
  faster than the time threshold, so the queries see an empty table.

  The proper fix is to add flush_record_buffer() calls in each bench
  after batch insert (a per-bench refactor).

  This shim is a TONIGHT-ONLY wrapper that monkey-patches
  MemoryStore.query_similar and the recall pipeline entry points to
  flush the record/edge/event buffers first. The patches affect the
  bench-process only; the production daemon (separate process) is
  unaffected. The patches do not change persistent data, only the
  flush timing.

Usage:
    .venv/bin/python bench/_night_runner.py <bench_module> [args...]

Example:
    IAI_MCP_STORE=/tmp/bench-1 IAI_MCP_CRYPTO_PASSPHRASE=disposable \\
      .venv/bin/python bench/_night_runner.py verbatim
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

# When invoked as `python bench/_night_runner.py …`, Python puts only
# `bench/` on sys.path. Add the worktree root + src/ so `from bench import …`
# and `import iai_mcp` both resolve. Canonical bench shim form (matches
# tests/test_bench_worktree_resolution.py contract: `_SRC_PATH =...` +
# `if _SRC_PATH not in sys.path:` at module level, BEFORE any iai_mcp/bench
# import).
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)


# Static dispatch table: every supported bench is mapped to a thunk that
# performs an EXPLICIT literal-name import. No dynamic `importlib.import_module`
# call anywhere, so semgrep CWE-706 cannot fire and there is no way for an
# attacker-controlled argv to load arbitrary code.
def _load_verbatim() -> ModuleType:
    from bench import verbatim  # noqa: PLC0415
    return verbatim


def _load_tokens() -> ModuleType:
    from bench import tokens  # noqa: PLC0415
    return tokens


def _load_community_pipeline_perf() -> ModuleType:
    from bench import community_pipeline_perf  # noqa: PLC0415
    return community_pipeline_perf


def _load_neural_map() -> ModuleType:
    from bench import neural_map  # noqa: PLC0415
    return neural_map


def _load_trajectory() -> ModuleType:
    from bench import trajectory  # noqa: PLC0415
    return trajectory


def _load_personal_fact_drift() -> ModuleType:
    from bench import personal_fact_drift  # noqa: PLC0415
    return personal_fact_drift


def _load_memory_footprint() -> ModuleType:
    from bench import memory_footprint  # noqa: PLC0415
    return memory_footprint


_BENCH_DISPATCH = {
    "verbatim": _load_verbatim,
    "tokens": _load_tokens,
    "community_pipeline_perf": _load_community_pipeline_perf,
    "neural_map": _load_neural_map,
    "trajectory": _load_trajectory,
    "personal_fact_drift": _load_personal_fact_drift,
    "memory_footprint": _load_memory_footprint,
}


def _apply_patches() -> None:
    """Monkey-patch query paths to flush record/edge/event buffers first."""
    from iai_mcp.store import MemoryStore, flush_record_buffer, flush_edge_buffer
    from iai_mcp.events import flush_event_buffer
    from iai_mcp import retrieve

    _original_query_similar = MemoryStore.query_similar
    _original_get = MemoryStore.get
    _original_recall = retrieve.recall

    def _flush_all(store: MemoryStore) -> None:
        try:
            flush_record_buffer(store)
        except Exception:
            pass
        try:
            flush_edge_buffer(store)
        except Exception:
            pass
        try:
            flush_event_buffer(store)
        except Exception:
            pass

    def patched_query_similar(self: MemoryStore, *args, **kwargs):
        _flush_all(self)
        return _original_query_similar(self, *args, **kwargs)

    def patched_get(self: MemoryStore, *args, **kwargs):
        _flush_all(self)
        return _original_get(self, *args, **kwargs)

    def patched_recall(*args, **kwargs):
        store = kwargs.get("store") if "store" in kwargs else (args[0] if args else None)
        if store is not None:
            _flush_all(store)
        return _original_recall(*args, **kwargs)

    MemoryStore.query_similar = patched_query_similar
    MemoryStore.get = patched_get
    retrieve.recall = patched_recall


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(
            "usage: bench/_night_runner.py <bench_name> [args...]\n"
            f"supported: {sorted(_BENCH_DISPATCH)}",
            file=sys.stderr,
        )
        return 2

    bench_name = argv[0]
    bench_args = argv[1:]

    loader = _BENCH_DISPATCH.get(bench_name)
    if loader is None:
        print(
            f"unsupported bench {bench_name!r}; "
            f"supported: {sorted(_BENCH_DISPATCH)}",
            file=sys.stderr,
        )
        return 2

    _apply_patches()

    mod = loader()

    if not hasattr(mod, "main"):
        print(f"bench {bench_name} has no main() — cannot drive", file=sys.stderr)
        return 2

    rc = mod.main(bench_args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
