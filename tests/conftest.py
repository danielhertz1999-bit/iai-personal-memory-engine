"""Project-wide pytest fixtures for the IAI-MCP test suite.

The file-based crypto key migration removed the keyring backend
from `iai_mcp.crypto.CryptoKey.get_or_create()`. Pre-existing tests that
exercised the daemon, store, events, recall, and CLI paths relied on the
keyring auto-fallback to source the encryption key in test environments.
After that migration, the runtime path is **file → passphrase env → error**
with no keyring fallback, so those tests now hit `CryptoKeyError` unless
either the file or the passphrase is set.

This module's autouse fixture sets `IAI_MCP_CRYPTO_PASSPHRASE` to a fixed
test passphrase for every test session, restoring the deterministic
`derive_key_from_passphrase(...)` path that the test suite expects.
Production behavior is unaffected — the production daemon never sets
this env var and instead reads the 32-byte file at `{IAI_MCP_STORE}/.crypto.key`
written by `iai-mcp crypto migrate-to-file` or `iai-mcp crypto init`.

The dedicated file-backend tests in `tests/test_crypto_file_backend.py`
override this fixture per-test by clearing the env var or by writing an
explicit `.crypto.key` file in their `tmp_path` fixtures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make repo-root-level packages (e.g. `scripts/`) importable from tests.
# pyproject's [tool.pytest.ini_options] pythonpath is `["src"]` only; tests
# that import top-level helper packages need the repo root prepended here.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make the tests directory importable so shared test-only helper modules
# (e.g. `_recall_helpers`, `test_store`) resolve by bare name.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Re-export the daemon-independent recall seeding + structural-cache helpers so
# both the in-process gate and the real-subprocess gate share ONE definition.
# These build the on-disk structural cache (NOT a monkeypatch), so a subprocess
# running the real recall path reuses the SAME cache layout.
from _recall_helpers import (  # noqa: E402,F401
    _deterministic_vec,
    _make_gold_record,
    _populate_store,
    _prime_structural_cache,
    _random_vec,
    UUID_HUB,
    UUID_INTER,
    UUID_SEED,
    UUID_TWO_HOP,
    UUID_TWO_HOP_SURFACE,
)


_TEST_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30-phase-07.10"


@pytest.fixture(autouse=True)
def _hermetic_default_paths(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """Redirect-by-default: every test runs against a per-test tmp store root.

    Defined FIRST in the autouse block so it orders before any other autouse
    fixture (and before any store-opening fixture): it depends only on
    ``tmp_path_factory`` + ``monkeypatch``, never on a fixture that opens a
    store. It points HOME + IAI_DAEMON_SOCKET_PATH at a fresh tmp ``.iai-mcp``
    dir AND ``monkeypatch.setattr``s the frozen import-time default constants to
    that tmp dir, so a bare ``MemoryStore()`` / default ``HippoDB()`` resolves
    to tmp, never to the operator's real home store.

    Deliberately does NOT set IAI_MCP_STORE: setting it would split
    ``store.root`` from the HippoDB dir on ``path=`` tests, because the two
    resolvers differ in precedence (env > path vs path > env). With env unset
    and only the defaults redirected, an explicit ``path=`` wins consistently
    at both layers, and a bare store hits the redirected tmp default.

    Tests with their own env/setattr (e.g. an explicit IAI_MCP_STORE) override
    this fixture: monkeypatch is last-write-wins within a test, so a later
    setattr/setenv supersedes these redirects.
    """
    base = tmp_path_factory.mktemp("iai-hermetic")
    fake_root = base / ".iai-mcp"
    fake_root.mkdir(parents=True, exist_ok=True)

    # Point the model-cache env at the operator's real warm cache BEFORE
    # redirecting HOME. This keeps the ONLINE embedder path (and any tooling
    # that honours the HF env) on the warm bge-small model instead of a cold
    # tmp re-resolve. NOTE: the native offline branch uses the cache crate's
    # default, which reads $HOME (NOT this env), so the dedicated offline test
    # restores $HOME itself; these vars do not fix that branch. Read-only model
    # access; no operator memory/PII lives in that path. All three names are a
    # safe superset across crate versions. Located via the login database, not
    # $HOME, so the value is stable regardless of the HOME redirect below.
    from iai_mcp.hippo import _operator_home
    _real_cache = _operator_home() / ".cache" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(_real_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(_real_cache / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(_real_cache / "hub"))

    monkeypatch.setenv("HOME", str(base))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(fake_root / ".daemon.sock"))
    # Redirect the frozen DEFAULT constants (NOT IAI_MCP_STORE — see docstring).
    # Every frozen, home-derived default that resolves under the operator's real
    # ~/.iai-mcp is redirected to the tmp dir so no test can read/write the real
    # store through a default branch. Constants that are genuinely shadowed by an
    # env var the consumer checks first (cli.SOCKET_PATH via IAI_DAEMON_SOCKET_PATH)
    # or that resolve outside ~/.iai-mcp (cli.LAUNCHD_TARGET) are intentionally
    # left alone; the frozen-constant meta-test enforces this completeness.
    import iai_mcp.hippo as _hippo
    import iai_mcp.store as _store
    import iai_mcp.concurrency as _conc
    import iai_mcp.daemon_state as _ds
    import iai_mcp.lifecycle_state as _lifecycle_state
    import iai_mcp.cli as _cli
    import iai_mcp.lifecycle_event_log as _lel
    import iai_mcp.capture_queue as _cq
    import iai_mcp.lifecycle as _lifecycle
    import iai_mcp.daemon as _daemon
    import iai_mcp.crypto as _crypto
    import iai_mcp.backup as _backup
    monkeypatch.setattr(_hippo, "_DEFAULT_IAI_ROOT", fake_root, raising=False)
    monkeypatch.setattr(_store, "DEFAULT_STORAGE_PATH", fake_root, raising=False)
    monkeypatch.setattr(_conc, "SOCKET_PATH", fake_root / ".daemon.sock", raising=False)
    monkeypatch.setattr(_ds, "STATE_PATH", fake_root / ".daemon-state.json", raising=False)
    monkeypatch.setattr(
        _lifecycle_state, "LIFECYCLE_STATE_PATH",
        fake_root / "lifecycle_state.json", raising=False,
    )
    monkeypatch.setattr(_cli, "LOCK_PATH", fake_root / ".lock", raising=False)
    monkeypatch.setattr(
        _cli, "STATE_PATH", fake_root / ".daemon-state.json", raising=False,
    )
    monkeypatch.setattr(_lel, "DEFAULT_LOG_DIR", fake_root / "logs", raising=False)
    monkeypatch.setattr(_cq, "DEFAULT_QUEUE_DIR", fake_root / "pending", raising=False)
    monkeypatch.setattr(
        _lifecycle, "DEFAULT_LOCK_PATH", fake_root / ".lifecycle.lock", raising=False,
    )
    monkeypatch.setattr(
        _daemon, "SESSION_START_CACHE_PATH",
        fake_root / ".session-start-payload.cached.md", raising=False,
    )
    monkeypatch.setattr(_crypto, "_DEFAULT_STORE_ROOT", fake_root, raising=False)
    monkeypatch.setattr(_backup, "DEFAULT_STORE_PATH", str(fake_root), raising=False)
    yield fake_root


@pytest.fixture(autouse=True)
def _clear_autoflush_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety-net: clear IAI_MCP_TEST_NO_AUTOFLUSH at the start of every test.

    Prevents a leaked value from a prior test from silently disabling the
    autoflush wrapper, which would cause store.insert() to leave records in
    the in-memory buffer and make store.get() return None for the next test.
    Tests that explicitly need the opt-out state set it themselves via their
    own monkeypatch.setenv call, which takes effect after this fixture runs.
    """
    monkeypatch.delenv("IAI_MCP_TEST_NO_AUTOFLUSH", raising=False)


@pytest.fixture(autouse=True)
def _crypto_passphrase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set IAI_MCP_CRYPTO_PASSPHRASE for every test unless already set.

    Tests that need to assert the absent-passphrase / missing-key error
    path can still call `monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE",
    raising=False)` inside the test body to override this default.
    """
    if "IAI_MCP_CRYPTO_PASSPHRASE" not in os.environ:
        monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_PASSPHRASE)


# Buffered-write triage: the buffered-write extension to
# RECORDS + EDGES tables means `store.insert(rec)` no longer flushes to
# the store immediately — rows live in `_record_buffer` until the daemon's
# WAKE / periodic-tick / shutdown hooks call `flush_record_buffer`.
# Pre-buffering tests that did
#
#     store.insert(rec)
#     df = store.db.open_table(RECORDS_TABLE).to_pandas()
#
# and then `df[df["id"] == ...].iloc[0]` started failing en masse with
# `IndexError: single positional indexer is out-of-bounds` because the
# row never made it to disk before the read.
#
# The autouse fixture below patches `MemoryStore.insert`, `boost_edges`,
# and `add_contradicts_edge` to call their respective flush helpers
# immediately after the buffered append.  This restores the pre-buffering
# observable behaviour for the test suite while leaving production code
# completely untouched.
#
# Tests that explicitly need to observe the buffered (un-flushed) state
# — the buffer tests, the SIGKILL test, the events-buffer
# tests, etc. — set `IAI_MCP_TEST_NO_AUTOFLUSH=1` in their own monkeypatch
# block to override this fixture.
_AUTOFLUSH_OPT_OUT_ENV = "IAI_MCP_TEST_NO_AUTOFLUSH"


@pytest.fixture(autouse=True)
def _autoflush_lance_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-flush RECORDS + EDGES buffers after every store-mutating call.

    See module docstring above.  Set ``IAI_MCP_TEST_NO_AUTOFLUSH=1``
    inside the test body to opt out (buffer-internals tests do this).
    The env-var check happens at call-time inside the wrapped methods so
    that per-file ``monkeypatch.setenv`` opt-outs take effect even if
    they're applied by a fixture that runs after this one.
    """
    try:
        from iai_mcp import store as _store_mod
    except Exception:  # noqa: BLE001 -- env without iai_mcp installed yet
        return

    MemoryStore = getattr(_store_mod, "MemoryStore", None)
    flush_record_buffer = getattr(_store_mod, "flush_record_buffer", None)
    flush_edge_buffer = getattr(_store_mod, "flush_edge_buffer", None)
    if (
        MemoryStore is None
        or flush_record_buffer is None
        or flush_edge_buffer is None
    ):
        return

    # Also flush buffered events so tests calling query_events() immediately
    # after store.insert() see events written with buffered=True.
    try:
        from iai_mcp.events import flush_event_buffer as _flush_event_buffer
    except Exception:  # noqa: BLE001
        _flush_event_buffer = None

    def _opt_out() -> bool:
        return os.environ.get(_AUTOFLUSH_OPT_OUT_ENV) == "1"

    _orig_insert = MemoryStore.insert

    def _insert_then_flush(self, *args, **kwargs):
        result = _orig_insert(self, *args, **kwargs)
        if _opt_out():
            return result
        try:
            flush_record_buffer(self)
            flush_edge_buffer(self)
            if _flush_event_buffer is not None:
                _flush_event_buffer(self)
        except Exception:  # noqa: BLE001 -- flush MUST NOT fail the test
            pass
        return result

    monkeypatch.setattr(MemoryStore, "insert", _insert_then_flush)

    _orig_boost = getattr(MemoryStore, "boost_edges", None)
    if _orig_boost is not None:
        def _boost_then_flush(self, *args, **kwargs):
            result = _orig_boost(self, *args, **kwargs)
            if _opt_out():
                return result
            try:
                flush_edge_buffer(self)
            except Exception:  # noqa: BLE001
                pass
            return result

        monkeypatch.setattr(MemoryStore, "boost_edges", _boost_then_flush)

    _orig_add_contradicts = getattr(MemoryStore, "add_contradicts_edge", None)
    if _orig_add_contradicts is not None:
        def _add_contradicts_then_flush(self, *args, **kwargs):
            result = _orig_add_contradicts(self, *args, **kwargs)
            if _opt_out():
                return result
            try:
                flush_edge_buffer(self)
            except Exception:  # noqa: BLE001
                pass
            return result

        monkeypatch.setattr(
            MemoryStore, "add_contradicts_edge", _add_contradicts_then_flush,
        )

    # Deferred-provenance: recall paths write provenance to a
    # deferred JSONL buffer first; tests reading `record.provenance`
    # directly after a recall see 0 entries until flush_deferred_provenance
    # drains the buffer.  Auto-flush right after every `defer_provenance`
    # call so any test path (whether it goes through `core.dispatch` or
    # imports `dispatch` into its own namespace) sees synchronous
    # provenance semantics — matching the contract in
    # tests/test_provenance.py ("Two recalls -> two new provenance
    # entries (reconsolidation never idempotent)").
    #
    # Previously this fixture patched `MemoryStore.memory_recall`, which
    # was a silent no-op because `MemoryStore` never had such a method —
    # the actual entry point is `iai_mcp.core.dispatch`.  Patching the
    # `defer_provenance` function in `iai_mcp.provenance_buffer` is the
    # only attachment point that catches every call site (pipeline.py
    # imports it dynamically, so `pipeline.defer_provenance` rebinds on
    # every recall).  Same opt-out env var applies — tests that assert
    # the deferred (un-flushed) buffer state set ``IAI_MCP_TEST_NO_AUTOFLUSH=1``.
    try:
        from iai_mcp import provenance_buffer as _prov_buf_mod
    except Exception:  # noqa: BLE001
        _prov_buf_mod = None

    if _prov_buf_mod is not None:
        _orig_defer = _prov_buf_mod.defer_provenance
        _orig_flush = _prov_buf_mod.flush_deferred_provenance

        def _defer_then_flush(store, entries):
            result = _orig_defer(store, entries)
            if _opt_out():
                return result
            try:
                _orig_flush(store)
            except Exception:  # noqa: BLE001
                pass
            return result

        monkeypatch.setattr(_prov_buf_mod, "defer_provenance", _defer_then_flush)


# Opt-in --runslow flag for subprocess-heavy tests (bench-shim resolution).
# Default: slow-marked tests are skipped so the default pytest run stays fast.
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (subprocess-heavy bench-shim resolution checks)",
    )
    parser.addoption(
        "--perf",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.perf (wall-clock latency benches, out of the default gate)",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run @pytest.mark.live integration tests (real daemon subprocess; out of the default correctness gate)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Three INDEPENDENT opt-in gates. Do NOT short-circuit one on the other:
    # a single early `return` for --runslow would let --perf/--live tests run
    # whenever --runslow is passed (and vice-versa). Each marker is gated by
    # its own flag so the three are fully orthogonal.
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--perf"):
        skip_perf = pytest.mark.skip(reason="need --perf to run wall-clock bench")
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="need --live to run the real-daemon E2E gate")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


# ---------------------------------------------------------------- mosaicsigma test adapter
#
# Adapter for σ tests: ``fast_sigma`` accepts ``MemoryGraph`` only. Legacy
# tests that build oracle fixtures via networkx generators (e.g.
# ``nx.connected_watts_strogatz_graph``, ``nx.gnm_random_graph``) feed those
# into this adapter before calling ``fast_sigma``.
#
# Placed in ``tests/conftest.py`` (not a per-file local helper) so any phase-50
# test file can import it via ``from tests.conftest import _nx_graph_to_memory_graph``.
# ---------------------------------------------------------------------------
# Hermetic tmp-store fixture for daemon-decoupling tests.
#
# All daemon-decoupling tests MUST use this fixture instead of
# constructing MemoryStore directly on the default path.  The fixture:
#   - Points IAI_MCP_STORE at a fresh tmp subdirectory (never ~/.iai-mcp).
#   - Points IAI_DAEMON_SOCKET_PATH at a non-existent path so every daemon
#     socket probe in the process under test fails immediately.
#   - Sets HOME to tmp_path so Path.home()/.iai-mcp never resolves to the
#     live home directory (guards capture.read_pending_live_events in tests
#     that call iai_cli subcommands or cmd_last/cmd_recall).
#   - Yields the store root Path; the crypto passphrase is already set by
#     _crypto_passphrase_env above.
#
# Test data: generic "User" identity string and generic text only.
# NO real names, NO machine paths, NO PII.
# ---------------------------------------------------------------------------


@pytest.fixture
def hermetic_store(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Return a hermetic store root isolated to tmp_path.

    Sets IAI_MCP_STORE, IAI_DAEMON_SOCKET_PATH, and HOME to tmp
    subdirectories so no test using this fixture can contact the live
    ~/.iai-mcp store or the running daemon.
    """
    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    dead_socket = tmp_path / "no-such.sock"   # intentionally absent
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(dead_socket))
    yield store_root


# ---------------------------------------------------------------------------
# Module-level singleton reset — keeps test ordering deterministic.
#
# Several modules carry process-wide mutable singletons that accumulate state
# across tests in a single-process pytest run.  Each singleton is correct in
# production (every real process starts clean), but cross-test leakage makes
# order-dependent failures.  This autouse fixture resets only the known
# offending singletons to their module-import initial values before each test.
# It is intentionally narrow: only the two modules and their specific globals
# listed below are touched.  Guard imports make the fixture a no-op in
# environments where the modules are not yet importable.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_singletons() -> None:
    """Reset process-wide mutable singletons before each test.

    Resets only the specific globals known to leak across tests:

    runtime_graph_cache:
      _current_generation  — monotonic epoch stamped by the nightly rebuild.
      _rebuild_timestamp_override — transient write-path flag.
      dirty counter  — record-mutation counter (reset via public helper).

    semantic_recall:
      _WARM_LOCAL_STORE    — cached local MemoryStore handle for the
                             daemon-independent structural path.
    """
    try:
        import iai_mcp.runtime_graph_cache as _rgc
        with _rgc._GEN_LOCK:
            _rgc._current_generation = 0
            _rgc._rebuild_timestamp_override = ""
        _rgc.reset_dirty_counter()
    except Exception:  # noqa: BLE001 -- not yet installed in some test envs
        pass

    try:
        import iai_mcp.semantic_recall as _sr
        _sr._WARM_LOCAL_STORE = None
    except Exception:  # noqa: BLE001 -- not yet installed in some test envs
        pass


def _nx_graph_to_memory_graph(nx_g):
    """Copy an ``nx.Graph`` into a fresh ``MemoryGraph``.

    Builds a stable bijection from arbitrary ``nx`` node ids (ints or
    strings) to UUIDs, then materialises the resulting ``MemoryGraph``
    via ``add_node`` / ``add_edge``. Edge weights default to ``1.0`` when
    the source graph has no ``weight`` attribute.
    """
    from uuid import uuid4

    from iai_mcp.graph import MemoryGraph

    mg = MemoryGraph()
    node_to_uuid = {n: uuid4() for n in nx_g.nodes()}
    for _n, uid in node_to_uuid.items():
        mg.add_node(uid, community_id=None, embedding=[0.0] * 384)
    for u, v, data in nx_g.edges(data=True):
        w = 1.0
        try:
            w = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        mg.add_edge(node_to_uuid[u], node_to_uuid[v], weight=w)
    return mg
