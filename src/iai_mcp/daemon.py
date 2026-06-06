"""IAI-MCP Sleep Daemon main entry point.

Guards:
- Human-first: the daemon NEVER blocks the awake read/write path. The legacy
  in-process exclusive gate was removed; contention now lives on the storage
  lock (the awake read/write lock) and the lifecycle marker owns single-machine
  singleton ownership. The per-tick maintenance body runs only lightweight
  per-tick work and the heavy consolidation cycle yields cooperatively.
- User consent: the daemon NEVER initiates sleep mode without explicit user
  consent; the consent gate lives in bedtime.py. Distinct from the structural
  dispatcher/FSM-isolation invariant declared in socket_server.py.
- Zero API cost: this module does NOT reference the paid-API env var;
  claude_cli.py is wired with env scrubbed at subprocess creation.
- Clean uninstall via signal.SIGTERM -> shutdown event -> task cancel +
  state persisted + the lifecycle marker released. launchd/systemd stop this
  daemon cleanly.
- Literal preservation -- the daemon never assigns to record.literal_surface.
  Called modules (sleep.py / schema.py) respect literal preservation by design.
- S5 audit runs read-only; spawned as an independent task alongside the
  scheduler so it continues even when the scheduler is blocked on a heavy op.

The scheduler tick loop only emits `tick_error` events on exception; it never
crashes. The per-tick maintenance body handles the empty-store shortcut,
quiet-window re-learn, bootstrap fallback, the N-cycle REM loop via
`dream.run_rem_cycle`, FSM transitions, and pending_digest accumulation.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import faulthandler
import json
import logging
import os
import resource
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

from iai_mcp import s4
from iai_mcp.concurrency import serve_control_socket  # noqa: F401 -- re-exported here for the test suite; the function lives in concurrency.py
from iai_mcp.daemon_state import load_state, save_state
from iai_mcp.dream import run_rem_cycle
from iai_mcp.events import (
    DAEMON_MEMORY_PRESSURE_KILL,
    DAEMON_WATCHDOG_NEEDS_OPERATOR,
    DAEMON_WEDGE_KILL,
    write_event,
)
from iai_mcp.identity_audit import continuous_audit
from iai_mcp.quiet_window import (
    BUCKET_COUNT,
    BUCKET_MINUTES,
    learn_quiet_window,
    should_bootstrap_trigger,
    should_relearn,
)
from iai_mcp.hippo import AccessMode
from iai_mcp.lock_protocol import cleanup_stale_consolidation_intent
from iai_mcp.native_guard import _require_native
from iai_mcp.sleep_wal import SleepWAL
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore
from iai_mcp.tz import load_user_tz

# ---------------------------------------------------------------------------
# State machine constants
# ---------------------------------------------------------------------------

STATE_WAKE: str = "WAKE"
STATE_TRANSITIONING: str = "TRANSITIONING"
STATE_SLEEP: str = "SLEEP"
STATE_DREAMING: str = "DREAMING"

# Valid FSM edges. DREAMING must return via SLEEP on wake.
VALID_TRANSITIONS: dict[str, set[str]] = {
    STATE_WAKE: {STATE_TRANSITIONING},
    STATE_TRANSITIONING: {STATE_SLEEP, STATE_WAKE},
    STATE_SLEEP: {STATE_DREAMING, STATE_WAKE},
    STATE_DREAMING: {STATE_SLEEP},
}

# Scheduler tick cadence (seconds). Light tick every 30s; hourly / 3h / 24h
# periodic work is gated inside _tick_body by last-ran timestamps.
TICK_INTERVAL_SEC: int = 30

# Default cycle count per quiet window (biologically typical 4-5).
DEFAULT_CYCLE_COUNT: int = 4

# Hourly cadence for the S4 offline pass (FSRS wall-clock decay + viability scan).
# Matches the sigma snapshot cadence in identity_audit so the daemon has a single
# coherent "hourly heartbeat" of diagnostics.
S4_OFFLINE_INTERVAL_SEC: int = 60 * 60

# Startup grace period before the FIRST iteration of `_s4_offline_loop`.
# The S4 offline pass walks the full graph and on cold caches calls
# `runtime_graph_cache.save -> json.dumps`, materialising a multi-GB
# intermediate string. Default = S4_OFFLINE_INTERVAL_SEC (1h, matching
# steady-state cadence). Set to 0 for tests / explicit warm-start.
# Env override: IAI_MCP_S4_FIRST_ITER_GRACE_SEC.
S4_FIRST_ITER_GRACE_SEC: float = float(
    os.environ.get("IAI_MCP_S4_FIRST_ITER_GRACE_SEC", str(S4_OFFLINE_INTERVAL_SEC)),
)

# Precache for SessionStart hook.
# Cached recall payload written once per REM-loop completion so the
# SessionStart hook reads a file instead of dispatching a JSON-RPC
# call into core (which blocks on the exclusive store lock during
# DREAMING). Path mirrors the hook's $HOME/.iai-mcp/.session-start-payload.cached.md.
SESSION_START_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"
from iai_mcp.session import SESSION_START_CACHE_MAX_CHARS  # noqa: E402 -- placed after PATH constant for readability

# Window inside which an MCP touch / open connection means the
# daemon should defer the next sleep_pipeline chunk (interrupt).
INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC: float = 30.0


# ---------------------------------------------------------------------------
# Boot health check — HippoDB integrity audit (replaces the former backend's
# startup compaction which is not applicable to the SQLite/hnswlib backend).
# ---------------------------------------------------------------------------


def _hippo_health_check_on_boot(store) -> dict[str, int | str]:
    """Verify HippoDB opened cleanly + hnswlib active count matches SQLite.

    HippoDB.__init__ already handles index rebuild on divergence (the
    integrity check + rebuild path lives inside the Hippo backend
    itself). This function only reads the resulting state and produces
    an auditable summary the daemon emits as one boot event.

    Returns {"sqlite_count": int, "hnsw_active_count": int,
    "hnsw_raw_count": int, "action": str} where action is "ok" (counts
    match) or "divergence_at_boot" (HippoDB's internal rebuild path
    should have already fired — emit warning if not).

    The parity check uses ``len(store.db._label_map)`` (active-only,
    tombstoned records excluded). ``hnsw_raw_count`` is reported
    separately for diagnostic value but does NOT drive the
    ok/divergence decision (it includes soft-deleted slots).
    """
    try:
        db = store.db
        sqlite_count_row = db._conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()
        sqlite_count = int(sqlite_count_row[0]) if sqlite_count_row else 0
    except Exception as exc:
        return {
            "sqlite_count": -1,
            "hnsw_active_count": -1,
            "hnsw_raw_count": -1,
            "action": "sqlite_count_failed",
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    # Active-count parity (M-05): use _label_map length, not
    # hnswlib.get_current_count(). The label_map is decremented on
    # tombstone path, so its length matches the active set in SQLite.
    try:
        active_label_count = int(len(db._label_map))
    except Exception:
        active_label_count = -1
    try:
        hnsw_raw_count = int(db._hnsw.get_current_count())
    except Exception:
        hnsw_raw_count = -1
    action = (
        "ok"
        if sqlite_count == active_label_count
        else "divergence_at_boot"
    )
    return {
        "sqlite_count": sqlite_count,
        "hnsw_active_count": active_label_count,
        "hnsw_raw_count": hnsw_raw_count,
        "action": action,
    }


# ---------------------------------------------------------------------------
# FD-limit hardening — raise the soft RLIMIT_NOFILE at boot
# ---------------------------------------------------------------------------

#: Default target for the soft RLIMIT_NOFILE floor.  A socket-serving
#: daemon that opens one fd per connected client needs headroom well above
#: the tiny launchd default.  8192 is conservative and safe on all modern
#: macOS / Linux kernels (typical hard limit is in the millions or
#: RLIM_INFINITY on macOS).  Override via IAI_MCP_DAEMON_NOFILE_FLOOR.
_DAEMON_NOFILE_FLOOR_DEFAULT: int = 8192


def _raise_fd_limit() -> None:
    """Raise the process soft RLIMIT_NOFILE to a sane floor at boot.

    Reads the current (soft, hard) pair and attempts to set soft to
    ``max(current_soft, floor)`` where ``floor`` comes from the env var
    ``IAI_MCP_DAEMON_NOFILE_FLOOR`` (default 8192).  The target is always
    clamped to the OS hard limit so setrlimit never receives an
    out-of-range value.

    When the OS hard limit is RLIM_INFINITY (common on macOS) the
    effective ceiling is the floor itself — requesting an infinite soft
    limit would silently error on some kernels.

    Failure is non-fatal: any OSError / ValueError from setrlimit is
    logged at DEBUG level and boot continues without interruption.
    """
    try:
        floor = int(
            os.environ.get("IAI_MCP_DAEMON_NOFILE_FLOOR", _DAEMON_NOFILE_FLOOR_DEFAULT)
        )
    except (TypeError, ValueError):
        floor = _DAEMON_NOFILE_FLOOR_DEFAULT

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return  # getrlimit failed; nothing we can do

    # When hard == RLIM_INFINITY treat the floor as the effective ceiling
    # so we never pass an astronomically large value to setrlimit.
    effective_hard = hard if hard != resource.RLIM_INFINITY else floor

    target = min(max(soft, floor), effective_hard)
    if target <= soft:
        return  # already at or above the floor; nothing to do

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        log.debug("daemon_fd_limit_raised soft=%d->%d hard=%d", soft, target, hard)
    except (OSError, ValueError) as exc:
        log.debug("daemon_fd_limit_raise failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# WAKE -> DROWSY drain edge helpers
# ---------------------------------------------------------------------------


def _should_drain_on_drowsy_edge(prev, current) -> bool:
    """True iff this is the edge into DROWSY (prev=WAKE, current=DROWSY)."""
    from iai_mcp.lifecycle_state import LifecycleState as _L
    return prev is _L.WAKE and current is _L.DROWSY


def _run_drowsy_drain(store, *, drain_fn, write_event_fn) -> None:
    """Run drain and emit one bookkeeping event.

    Writes ``deferred_drain_drowsy`` only when work was done; on exception
    swallows and writes ``deferred_drain_failed`` with ``phase='drowsy'``.
    Silent on zero-work to avoid log noise.
    """
    try:
        result = drain_fn(store)
    except Exception as e:  # noqa: BLE001 -- lifecycle_tick MUST NOT crash
        log.warning("drowsy drain failed: %s", e, exc_info=True)
        try:
            write_event_fn(
                store,
                "deferred_drain_failed",
                {"error": str(e)[:200], "phase": "drowsy"},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("failed to write deferred_drain_failed event: %s", e)
        return
    if not isinstance(result, dict):
        return
    if result.get("files_drained") or result.get("files_failed"):
        try:
            write_event_fn(
                store,
                "deferred_drain_drowsy",
                result,
                severity="info",
            )
        except Exception:  # noqa: BLE001 -- event write non-critical
            log.debug("failed to write deferred_drain_drowsy event")


def _kick_drowsy_rgc_rebuild(store) -> None:
    """Kick a non-blocking background graph-cache rebuild after a DROWSY-edge
    invalidate.

    Mirrors the boot-preload flag-not-gate pattern: returns immediately while
    a daemon-thread worker rebuilds the cache and sets
    ``runtime_graph_cache.rebuild_ready`` in a ``finally``.  Recall is NEVER
    blocked on this rebuild (flag-not-gate — daemon is never a gatekeeper).
    On rebuild failure the cache stays cold and recall degrades gracefully to
    the recency floor; the Event still sets so waiters never hang.

    The worker calls the rebuild via the module attribute
    ``runtime_graph_cache._rebuild_and_save_rgc`` (NOT a name bound at
    import time) so any test monkeypatch on that attribute reaches the worker.

    Safe to call from a plain sync context (no running event loop required).
    """
    import threading as _threading

    def _bg() -> None:
        try:
            import iai_mcp.runtime_graph_cache as _rgc
            _rgc._rebuild_and_save_rgc(store)
        except Exception:  # noqa: BLE001 -- best-effort; cache stays cold on failure
            log.debug("drowsy-edge graph-cache rebuild failed", exc_info=True)
        finally:
            try:
                import iai_mcp.runtime_graph_cache as _rgc
                _rgc.rebuild_ready.set()
            except Exception:  # noqa: BLE001
                log.debug("rebuild_ready.set() failed", exc_info=True)

    try:
        import iai_mcp.runtime_graph_cache as _rgc
        _rgc.rebuild_ready.clear()
    except Exception:  # noqa: BLE001
        log.debug("rebuild_ready.clear() failed", exc_info=True)

    _threading.Thread(target=_bg, daemon=True).start()


def _wake_hook_rebuild_if_cold(store) -> None:
    """Rebuild the graph cache only when it is cold after an interrupted cycle.

    Reads ``load_recall_structural(store)`` and calls
    ``runtime_graph_cache._rebuild_and_save_rgc(store)`` only when
    ``structural_source in {"cold_degrade", "last_good"}``.  When the cache
    is already overlay/normal it does nothing — no rebuild, no new generation
    stamp — to avoid extending the exclusive-lock window unnecessarily on an
    already-warm cache.

    Synchronous and safe to call from a plain test (no running event loop
    required).  Mirrors ``_kick_drowsy_rgc_rebuild`` / ``_run_drowsy_drain``
    at module level so this path has its own automated test.

    The rebuild call uses the module attribute so any test monkeypatch on
    ``runtime_graph_cache._rebuild_and_save_rgc`` is seen here.
    Any exception is swallowed (logged at debug) — this is best-effort and
    must never crash the SLEEP WAKE hook.
    """
    try:
        import iai_mcp.runtime_graph_cache as _rgc
        _, _, _, _src = _rgc.load_recall_structural(store)
        if _src in ("cold_degrade", "last_good"):
            _rgc._rebuild_and_save_rgc(store)
    except Exception:  # noqa: BLE001 -- best-effort, never crash the wake hook
        log.debug("wake-hook graph-cache rebuild failed", exc_info=True)


# ---------------------------------------------------------------------------
# State machine transitions (separated so tests can exercise directly)
# ---------------------------------------------------------------------------

def transition(state: dict, new_fsm: str) -> None:
    """Attempt the WAKE/TRANSITIONING/SLEEP/DREAMING edge.

    Raises ValueError when the edge is not in VALID_TRANSITIONS. Persists
    the new fsm_state + fsm_transition_at via save_state.
    """
    current = state.get("fsm_state", STATE_WAKE)
    allowed = VALID_TRANSITIONS.get(current, set())
    if new_fsm not in allowed:
        raise ValueError(
            f"Illegal transition {current} -> {new_fsm}; allowed: {sorted(allowed)}"
        )
    state["fsm_state"] = new_fsm
    state["fsm_transition_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


# ---------------------------------------------------------------------------
# Helpers used by _tick_body
# ---------------------------------------------------------------------------

def _store_is_empty(store: MemoryStore) -> bool:
    """Return True when the records table is empty (fast-path shortcut)."""
    try:
        return store.db.open_table("records").count_rows() == 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        log.debug("store empty check failed, assuming empty: %s", exc)
        return True


def _is_inside_window(
    window: tuple[int, int] | list | None,
    now: datetime,
    tz,
) -> bool:
    """Return True when the current local time falls inside the learned quiet
    window. Handles wrap-around across local midnight (e.g. 22:00 -> 06:00)."""
    if not window:
        return False
    try:
        start, duration = int(window[0]), int(window[1])
    except (TypeError, ValueError, IndexError):
        return False
    if duration <= 0:
        return False
    now_local = now.astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // BUCKET_MINUTES
    end = (start + duration) % BUCKET_COUNT
    if start < end:
        return start <= cur_bucket < end
    # Wrap-around (e.g. start=44 (22:00), duration=16, end=(44+16)%48=12 (06:00))
    return cur_bucket >= start or cur_bucket < end


# ---------------------------------------------------------------------------
# The legacy in-process C1 yield helper (and the MCP-recent-activity window)
# were removed. The lifecycle state machine now governs yield: when wrapper
# heartbeats are FRESH the daemon is in WAKE state and the sleep pipeline is
# never run; SLEEP-state work is bounded-deferred via the lifecycle tick's
# `interrupt_check`. The per-tick maintenance body therefore runs only
# lightweight work, and contention on the awake read/write path lives on the
# storage lock; consolidation cycles trigger only inside the learned quiet
# window.
# ---------------------------------------------------------------------------


def _update_pending_digest(state: dict, cycle_result: dict) -> None:
    """Accumulate per-cycle outputs into the morning digest."""
    digest = state.get("pending_digest") or {
        "rem_cycles_completed": 0,
        "episodes_processed": 0,
        "schemas_induced_tier0": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out_cycles": 0,
    }
    digest["rem_cycles_completed"] = int(digest.get("rem_cycles_completed", 0)) + 1
    digest["episodes_processed"] = int(digest.get("episodes_processed", 0)) + int(
        cycle_result.get("summaries_created", 0) or 0
    )
    digest["schemas_induced_tier0"] = int(digest.get("schemas_induced_tier0", 0)) + int(
        cycle_result.get("schema_candidates", 0) or 0
    )
    if cycle_result.get("claude_call_used"):
        digest["claude_call_used"] = True
        digest["main_insight_text"] = cycle_result.get("main_insight_text")
    if cycle_result.get("timed_out"):
        digest["timed_out_cycles"] = int(digest.get("timed_out_cycles", 0)) + 1
    state["pending_digest"] = digest


# Precache writer.
# Builds the same session-start payload core.dispatch would produce, renders
# it as markdown via session.format_payload_as_markdown, caps at
# SESSION_START_CACHE_MAX_CHARS, and atomically replaces the cache file.
# Forces wake_depth="standard" so the rendered output is non-empty even when
# the user's default profile is wake_depth="minimal" (which would produce
# only a compact handle and render to "").
# Calls the emit-free `_compose_session_start_payload` helper (NOT
# `assemble_session_start`) so this writer does not inject one synthetic
# `session_started` event per REM-loop completion into the events table.
def _write_session_start_cache(store, *, cache_path: Path = SESSION_START_CACHE_PATH) -> None:
    """Best-effort: write the session-start markdown payload to cache_path.

    Atomic via tmp-file + fsync + os.replace. Any exception is swallowed
    after a best-effort write_event(..., severity="warning"). This MUST
    NOT propagate into the REM loop.

    Uses `_compose_session_start_payload` (emit-free) so no
    `session_started` event is written per REM cycle. The live
    `core.dispatch` path continues to use `assemble_session_start` and
    still emits one event per real session.
    """
    try:
        from iai_mcp import retrieve
        from iai_mcp.session import (
            _compose_session_start_payload,
            format_payload_as_markdown,
        )

        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = _compose_session_start_payload(
            store,
            assignment,
            rc,
            session_id="precache",
            profile_state={"wake_depth": "standard"},
        )
        rendered = format_payload_as_markdown(payload)
        if not rendered:
            return  # nothing useful to cache
        if len(rendered) > SESSION_START_CACHE_MAX_CHARS:
            rendered = rendered[:SESSION_START_CACHE_MAX_CHARS]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, cache_path)
    except Exception as exc:  # noqa: BLE001 -- cache write MUST NOT crash the REM loop
        log.warning("session start cache write failed: %s", exc, exc_info=True)
        try:
            write_event(
                store,
                "session_start_cache_write_failed",
                {"error": str(exc)[:200]},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("failed to write session_start_cache_write_failed event")


# ---------------------------------------------------------------------------
# Scheduler tick body
# ---------------------------------------------------------------------------

async def _tick_body(
    store: MemoryStore,
    state: dict,
    *,
    mcp_socket: SocketServer | None = None,
) -> None:
    """One scheduler tick. Runs every TICK_INTERVAL_SEC (30s).

    Decision tree:
    0.5 Drain first_turn_pending entries older than 1 h. Runs FIRST so stale
        entries get cleared regardless of any yield/pause downstream. The
        helper is called with an explicit `now=` kwarg so its behaviour is
        fully driven by this tick's clock. Emits `first_turn_pending_expired`
        when entries are dropped.
    0. scheduler_paused -> skip immediately.
    1. Empty store -> short-circuit.
    2. Re-learn quiet window if 24h elapsed.
    3. Determine if we are inside the learned window OR the 2h-idle bootstrap
       OR a user_sleep_request / force_rem_request is pending. Otherwise return.
    4. Transition WAKE -> TRANSITIONING -> SLEEP.
    5. Loop up to DEFAULT_CYCLE_COUNT REM cycles via `run_rem_cycle`. Between
       cycles, check `force_wake_request`; on it, emit `daemon_yielded` and
       break.
    6. Transition SLEEP -> WAKE, persist state.

    Contention on the awake read/write path is owned by the storage lock; the
    legacy in-process exclusive gate was removed and REM cycles run only inside
    the learned quiet window. The `mcp_socket` kwarg is retained as
    accepted-and-ignored so existing tests keep working. Exceptions inside the
    REM loop surface as `rem_cycle_error` events emitted by dream.run_rem_cycle
    itself.
    """
    # --- Step 0.5: per-tick prune -----------------------------------------
    # Drain stale first_turn_pending entries (older than 1 h) on every tick.
    # Runs BEFORE any yield/pause/empty-store gate so stale entries clear
    # even when the rest of the tick would skip. Pure-in-memory walk +
    # at most one save_state + at most one event emit, all wrapped in
    # try/except so a malformed state never blocks the tick.
    #
    # Explicit `now=datetime.now(timezone.utc)` kwarg threads this tick's
    # clock into the helper; the helper does NOT call datetime.now itself
    # along this path, which keeps the function pure and trivially testable
    # by passing a fixed `NOW` directly.
    try:
        from iai_mcp.daemon_state import (
            FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
            prune_first_turn_pending,
        )

        state, dropped = prune_first_turn_pending(
            state, now=datetime.now(timezone.utc),
        )
        if dropped:
            try:
                await asyncio.to_thread(save_state, state)
            except (OSError, ValueError) as exc:  # noqa: BLE001 -- state save non-critical
                log.debug("save_state after prune failed: %s", exc)
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "first_turn_pending_expired",
                    {
                        "dropped_count": len(dropped),
                        "session_ids": dropped,
                        "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                        "phase": "tick",
                    },
                    severity="info",
                )
            except (OSError, RuntimeError) as exc:  # noqa: BLE001 -- event write non-critical
                log.debug("first_turn_pending_expired event write failed: %s", exc)
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        # Defense-in-depth: drain MUST NOT crash the tick. Auxiliary tick
        # steps swallow exceptions to preserve cooperative scheduling.
        log.warning("tick step 0.5 (prune first_turn_pending) failed", exc_info=True)

    # --- Step 0.6: S4 background contradiction scan --------------------------
    # Throttled to hourly (was per-tick — N+1 query_events was a CPU hog).
    try:
        _s4bg_ts = state.get("_last_s4bg_ts", "")
        _now_iso = datetime.now(timezone.utc).isoformat()
        _should_s4bg = not _s4bg_ts or (
            datetime.fromisoformat(_now_iso) - datetime.fromisoformat(_s4bg_ts)
        ).total_seconds() > 3600
        if _should_s4bg:
            from iai_mcp.s4 import s4_background_scan
            await asyncio.to_thread(s4_background_scan, store, 50)
            state["_last_s4bg_ts"] = _now_iso
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.debug("tick step 0.6 (s4_background_scan) failed", exc_info=True)

    # --- Step 0.7: Proactive internal foraging ----------------------------
    # Detect weak bridges between communities and create self-foraging edges.
    # Throttled to once per hour (graph rebuild is expensive).
    # SLEEP gate: skip foraging while the canonical lifecycle FSM is in SLEEP so
    # self_foraging boost_edges writes do not overlap the consolidation window.
    # self_foraging edges are WAKE-time exploratory bridges, not a
    # nightly-consolidation output; no learning is lost by skipping during SLEEP.
    # Reads lifecycle_state.json at call time (same file lifecycle_tick writes);
    # failure defaults to SKIP (conservative/safe).
    try:
        _forage_ts = state.get("_last_forage_ts", "")
        _now_iso = datetime.now(timezone.utc).isoformat()
        _should_forage = not _forage_ts or (
            datetime.fromisoformat(_now_iso) - datetime.fromisoformat(_forage_ts)
        ).total_seconds() > 3600
        if _should_forage:
            # SLEEP gate check: read canonical lifecycle state.
            _skip_foraging_in_sleep = False
            try:
                from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH, LifecycleState, load_state as _load_ls
                _ls_rec = await asyncio.to_thread(_load_ls, LIFECYCLE_STATE_PATH)
                _ls_current = _ls_rec.get("current_state", "")
                if _ls_current == LifecycleState.SLEEP.value:
                    _skip_foraging_in_sleep = True
            except Exception:
                # Cannot read lifecycle state — skip foraging conservatively.
                _skip_foraging_in_sleep = True
            if not _skip_foraging_in_sleep:
                from iai_mcp.foraging import forage_for_connections
                _foraged = await asyncio.to_thread(forage_for_connections, store, 3)
                state["_last_forage_ts"] = _now_iso
                if _foraged > 0:
                    await asyncio.to_thread(
                        write_event, store, "self_foraging_pass",
                        {"edges_created": _foraged}, severity="info",
                    )
            else:
                log.debug("tick step 0.7 (foraging) skipped: canonical FSM in SLEEP")
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.debug("tick step 0.7 (foraging) failed", exc_info=True)

    # --- Step 0.8: Periodic events-buffer flush --------------------------------
    # Catch buffers that accumulate slowly between WAKE transitions. The
    # WAKE-edge flush (in _post_sleep_tail) handles state-edge transitions;
    # this catches the steady-state tick window. Fires at most once per tick
    # + at most once per 5 s (should_flush_by_time gate).
    try:
        from iai_mcp.events import (
            _last_flush_at,
            flush_event_buffer,
            should_flush_by_time,
        )

        if should_flush_by_time(id(store), _last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_event_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("events buffer periodic flush skipped: %s", str(e)[:120])

    # Periodic records-buffer flush: drains the records buffer when it has
    # aged past should_flush_record_buffer_by_time's threshold. Fires at most
    # once per tick. Same fail-safe boundary as the events flush above.
    try:
        from iai_mcp.store import (
            _record_last_flush_at,
            flush_record_buffer,
            should_flush_record_buffer_by_time,
        )

        if should_flush_record_buffer_by_time(id(store), _record_last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_record_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("records buffer periodic flush skipped: %s", str(e)[:120])

    # Periodic edges-buffer flush: drains the edges buffer when aged past
    # should_flush_edge_buffer_by_time's threshold. Fires at most once per tick.
    try:
        from iai_mcp.store import (
            _edge_last_flush_at,
            flush_edge_buffer,
            should_flush_edge_buffer_by_time,
        )

        if should_flush_edge_buffer_by_time(id(store), _edge_last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_edge_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("edges buffer periodic flush skipped: %s", str(e)[:120])

    # --- Step -1: legacy in-process yield removed --------------------------
    # The legacy in-process HUMAN-FIRST yield is gone. The lifecycle state
    # machine + sleep pipeline supersede it: REM cycles only run inside the
    # learned quiet window, where MCP traffic is rare; the storage lock owns
    # contention on the awake read/write path if traffic arrives mid-cycle.

    # --- Step 0: scheduler_paused gate ------------------------------------
    if state.get("scheduler_paused") is True:
        try:
            await asyncio.to_thread(
                write_event,
                store,
                "daemon_tick_skipped",
                {"reason": "paused"},
                severity="info",
            )
        except (OSError, RuntimeError) as exc:
            log.debug("daemon_tick_skipped event write failed: %s", exc)
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "paused"
        try:
            await asyncio.to_thread(save_state, state)
        except (OSError, ValueError) as exc:
            log.debug("save_state (paused) failed: %s", exc)
        return

    # --- Step 1: empty store shortcut ---------------------------------------
    # Dispatched off-loop (to_thread) so a contended _conn_lock in a worker
    # thread cannot starve the event loop while this check runs.
    if await asyncio.to_thread(_store_is_empty, store):
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "empty_store"
        await asyncio.to_thread(save_state, state)
        return

    now = datetime.now(timezone.utc)
    try:
        tz = load_user_tz()
    except (OSError, ValueError, KeyError) as exc:
        # Config unreadable; fall back to UTC so we still run.
        log.debug("load_user_tz failed, using UTC: %s", exc)
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("UTC")

    # --- Step 2: re-learn quiet window every 24h ----------------------------
    last_learned_raw = state.get("quiet_window_learned_at")
    last_learned_dt: datetime | None = None
    if last_learned_raw:
        try:
            last_learned_dt = datetime.fromisoformat(last_learned_raw)
        except (TypeError, ValueError):
            last_learned_dt = None
    if should_relearn(last_learned_dt, now):
        try:
            window = await asyncio.to_thread(learn_quiet_window, store, now, tz)
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug("learn_quiet_window failed: %s", exc)
            window = None
        state["quiet_window"] = list(window) if window else None
        state["quiet_window_learned_at"] = now.isoformat()
        await asyncio.to_thread(save_state, state)

    # --- Steps 4-7 removed: consolidation now runs via the canonical lifecycle ---
    # The canonical lifecycle_tick -> _sleep_pipeline.run is the sole
    # consolidation driver. force_rem_request / user_sleep_request are
    # consumed by lifecycle_tick's FORCE_SLEEP dispatch (lifecycle.py).
    # The quiet-window / bootstrap gate is also removed: _tick_body now only
    # runs lightweight per-tick maintenance (prune, S4, foraging) so it always
    # runs regardless of window. Quiet-window enforcement for consolidation lives
    # in lifecycle_tick (idle detector + sleep_eligible check).

    # Tick completed: update state timestamp.
    state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
    try:
        await asyncio.to_thread(save_state, state)
    except (OSError, ValueError) as exc:
        log.debug("save_state after tick failed: %s", exc)


async def _scheduler_tick(
    store: MemoryStore,
    state: dict,
    *,
    tick_body: Callable[..., Awaitable[None]] | None = None,
    mcp_socket: SocketServer | None = None,
) -> None:
    """Run _tick_body every TICK_INTERVAL_SEC.

    An individual tick failure MUST NOT crash the daemon. We catch all
    exceptions, write a `tick_error` event (best-effort; even the event
    write is wrapped), and keep looping.

    When invoked from daemon.main(), mcp_socket is threaded through to
    _tick_body (accepted-and-ignored). Unit tests that pass a custom tick_body
    keep working — both the built-in _tick_body and a custom tick_body callable
    are invoked with keyword-only mcp_socket, with a 2-arg fallback for callables
    that do not accept the keyword.
    """
    body = tick_body or _tick_body
    while True:
        try:
            await body(store, state, mcp_socket=mcp_socket)
        except TypeError:
            # A custom tick_body callable may not accept the keyword-only
            # mcp_socket arg. Fall back to the 2-arg form so existing tests
            # keep passing without modification.
            try:
                await body(store, state)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 -- daemon tick boundary
                log.warning("tick failed (legacy body): %s", exc, exc_info=True)
                try:
                    write_event(
                        store,
                        "tick_error",
                        {"error": str(exc), "type": type(exc).__name__},
                        severity="warning",
                    )
                except Exception:  # noqa: BLE001 -- event write inside boundary guard
                    log.debug("tick_error event write failed")
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001 -- daemon must never die mid-tick
            log.warning("tick failed: %s", exc, exc_info=True)
            try:
                write_event(
                    store,
                    "tick_error",
                    {"error": str(exc), "type": type(exc).__name__},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("tick_error event write failed")
        try:
            await asyncio.sleep(TICK_INTERVAL_SEC)
        except asyncio.CancelledError:
            break


# ---------------------------------------------------------------------------
# S4 offline-pass loop (hourly viability scan, Warning 6)
# ---------------------------------------------------------------------------

async def _s4_offline_loop(store: MemoryStore, shutdown: asyncio.Event) -> None:
    """Hourly S4 viability scan -- contradictions, drift, stale goals, hit_rate.

    FSRS decay is applied by WALL-CLOCK elapsed time since last_reviewed (not
    per access count), so this loop only needs a wall-clock cadence; it does
    NOT iterate records or advance per-read counters. That keeps the loop
    cheap enough to run concurrent with other daemon work via SQLite/Hippo MVCC.

    A startup grace period delays the FIRST iteration so a freshly-spawned
    daemon does not immediately run the heavy S4 viability scan before
    draining deferred captures. Configured via S4_FIRST_ITER_GRACE_SEC
    (env IAI_MCP_S4_FIRST_ITER_GRACE_SEC). Cancellation semantics: if
    shutdown fires during the grace wait, the loop returns cleanly (no
    work performed, no exception).
    """
    if S4_FIRST_ITER_GRACE_SEC > 0:
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_FIRST_ITER_GRACE_SEC
            )
            # Shutdown fired during grace -- return without running S4.
            return
        except asyncio.TimeoutError:
            pass  # Grace elapsed; fall through to the regular loop.
    while not shutdown.is_set():
        try:
            await asyncio.to_thread(s4.run_offline_pass, store)
        except Exception as exc:  # noqa: BLE001 -- never die on offline-pass failure
            log.warning("S4 offline pass failed: %s", exc, exc_info=True)
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "s4_offline_pass_error",
                    {"error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("s4_offline_pass_error event write failed")
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_OFFLINE_INTERVAL_SEC
            )
            break
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------
# Activation cascade loop
# ---------------------------------------------------------------------------

# Poll cadence for the cascade loop. Short enough that a session_open event
# queued by the TS wrapper gets served within a few seconds; long enough
# that an idle loop doesn't spin the CPU.
HIPPEA_CASCADE_POLL_SEC: float = 5.0

# Minimum interval between cascade body executions.
# Default 60s = 12x the 5s poll cadence; gates heavy work without dropping
# `pending` flags. Env override IAI_MCP_HIPPEA_MIN_INTERVAL_SEC.
HIPPEA_CASCADE_MIN_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_HIPPEA_MIN_INTERVAL_SEC", "60.0"),
)

# Timestamp of the most recent cascade body completion (success or exception).
# Module-level mutable; the cascade loop declares `global _last_cascade_completed_at`
# to write. Ephemeral — daemon restart resets to 0.0.
_last_cascade_completed_at: float = 0.0

# Dedicated bounded ThreadPoolExecutor for cascade and maintenance work.
# Uses a separate pool (NOT the default asyncio to_thread pool) so the embed
# convoy — which saturates the default pool — cannot block cascade workers.
# Created by main() at daemon boot; patched in tests via the module attribute.
# max_workers=2: one active cascade + one spare for overlap during cooldown.
_cascade_executor: concurrent.futures.ThreadPoolExecutor | None = None


# ---------------------------------------------------------------------------
# CPU watchdog (OBSERVATION-ONLY — CPU is NEVER a kill signal)
# ---------------------------------------------------------------------------
# Polls own-process CPU every WATCHDOG_POLL_SEC; emits `daemon_cpu_overload`
# (severity=critical) on sustained > WATCHDOG_THRESHOLD_PERCENT for 2
# consecutive samples (= WATCHDOG_POLL_SEC * 2 seconds sustained), so a long
# blind period of high CPU with zero events cannot recur.
#
# This loop stays OBSERVATION-ONLY: high CPU means the daemon is making
# consolidation PROGRESS, so it is never killed here — no SIGTERM, no os.kill,
# no launchctl. The only side-effect is the event emit.
#
# Auto-recovery (a controlled self-kill -> launchd respawn) lives in the
# SEPARATE liveness/memory watchdog thread below, and fires ONLY on (a) a
# confirmed-unresponsive socket round-trip (a WEDGED loop that cannot serve a
# request) or (b) approaching-jetsam memory pressure (RSS leak cap, or sustained
# system memory-pressure with the daemon a significant RAM contributor). A busy
# daemon and a wedged daemon are distinguishable states; only the wedged one is
# killed. The self-kill is proven lossless before it was authorized.
WATCHDOG_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_POLL_SEC", "30.0"),
)
WATCHDOG_THRESHOLD_PERCENT: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_THRESHOLD_PERCENT", "50.0"),
)
WATCHDOG_EVENT_COOLDOWN_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_EVENT_COOLDOWN_SEC", "300.0"),
)
WATCHDOG_SAMPLE_WINDOW: int = 4

# ---------------------------------------------------------------------------
# Liveness + memory self-watchdog knobs (the auto-recovery watchdog).
# ---------------------------------------------------------------------------
# Distinct from the observation-only CPU watchdog above. This watchdog runs in
# a plain threading.Thread (NOT an asyncio task — it must keep running when the
# event loop wedges) and performs a controlled self-recovery on (a) a wedged
# loop or (b) approaching-jetsam memory pressure. Every knob has an env override
# + a code default. The RSS cap/floor byte values are the measured-peak-derived
# safe bounds (the cap sits well above the legitimate consolidation peak; the
# floor clears that peak yet stays strictly below the cap).
WATCHDOG_LIVENESS_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_LIVENESS_POLL_SEC", "30.0"),
)  # steady cadence, aligned to LIFECYCLE_TICK_INTERVAL_SEC
WATCHDOG_WARN_POLL_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_WARN_POLL_SEC", "7.0"),
)  # tightened cadence under sustained WARN pressure (in the ~5-10s band): at
# N=3 debounce this shortens the memory reaction window from 30s*3=90s to
# 7s*3=21s, well inside the jetsam margin, so a fast RSS balloon keeps its
# debounce window instead of losing it to the killer.
WATCHDOG_PROBE_TIMEOUT_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_PROBE_TIMEOUT_SEC", "5.0"),
)  # >> the ~3 ms healthy status reply; a read that exceeds this == wedged
WATCHDOG_FAILURE_DEBOUNCE_N: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_FAILURE_DEBOUNCE_N", "3"),
)  # consecutive failing ticks before a wedge/memory trigger acts
WATCHDOG_RSS_HARD_CAP_BYTES: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_RSS_HARD_CAP_BYTES", "2684354560"),
)  # 2.5 GiB — leak backstop; > the measured consolidation peak (~593 MB)
WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES", "1610612736"),
)  # 1.5 GiB — kill-on-pressure gate: the daemon must itself be a real RAM
# contributor (don't kill a small daemon when another process owns the RAM —
# pointless + an immediate re-WARN kill loop)
WATCHDOG_MAX_RECOVERIES: int = int(
    os.environ.get("IAI_MCP_WATCHDOG_MAX_RECOVERIES", "3"),
)  # circuit-breaker: after K self-kills in the window, STOP killing
WATCHDOG_RECOVERY_WINDOW_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_RECOVERY_WINDOW_SEC", "600.0"),
)  # the circuit-breaker window (wall-clock — see _load_recovery_timestamps)
WATCHDOG_COLD_START_GRACE_SEC: float = float(
    os.environ.get("IAI_MCP_WATCHDOG_COLD_START_GRACE_SEC", "600.0"),
)  # suppress the memory/leak trigger during the boot RSS ramp (the wedge
# trigger is NOT grace-covered — see _evaluate_watchdog)

# Pre-opened append-only fd for the LOCK-FREE kill breadcrumb. Opened once at
# watchdog-thread start (O_WRONLY|O_APPEND|O_CREAT, 0o600) so the kill path
# needs no allocation. None until the thread opens it.
_WATCHDOG_LOG_FD: int | None = None

# Pre-opened append-only fd for the lock-free forensic black-box dump.
# Opened alongside the breadcrumb fd in _liveness_watchdog. Separate file
# so the multi-line faulthandler output never corrupts the breadcrumb log
# that _load_recovery_timestamps parses line-by-line. None until opened.
_WATCHDOG_BLACKBOX_FD: int | None = None

# Episode-level gate for the forensic black box: True after the first dump
# fires in a failure episode; reset to False on a clean tick so the NEXT
# episode also dumps once. Module-level mutable; the watchdog thread writes
# it without a lock (only one watchdog thread ever runs at a time).
_WATCHDOG_BLACKBOX_EPISODE_FIRED: bool = False

# Env knob: set to "0" or "false" to disable the forensic black box.
# Default on. Intended for debugging / test environments where faulthandler
# output is not wanted.
_WATCHDOG_BLACKBOX_ENABLED: bool = (
    os.environ.get("IAI_MCP_WATCHDOG_BLACKBOX_ENABLED", "1").lower()
    not in ("0", "false", "no", "off")
)

# Boot lock-retry knobs: when the EXCLUSIVE store open races a dying
# predecessor's not-yet-released lock, bounded retry/backoff prevents a
# crash-loop. Both are env-tunable so they can be set very small in tests.
BOOT_LOCK_RETRY_ATTEMPTS: int = int(
    os.environ.get("IAI_MCP_BOOT_LOCK_RETRY_ATTEMPTS", "5"),
)  # max EXCLUSIVE-open attempts before giving up
BOOT_LOCK_RETRY_BACKOFF_SEC: float = float(
    os.environ.get("IAI_MCP_BOOT_LOCK_RETRY_BACKOFF_SEC", "0.5"),
)  # base sleep per attempt; delay grows linearly (base * attempt_number)

# Timestamp of the most recent overload event emit.
# Module-level mutable; `_cpu_watchdog_loop` declares `global` to write.
# Ephemeral — daemon restart resets to 0.0 so the first overload after
# restart can fire without waiting out a stale cooldown.
_last_overload_event_at: float = 0.0

# Monotonic boot timestamp; populated in main() after the daemon's
# wall-clock `daemon_started_at` stamp. Used by the watchdog to include
# `uptime_sec` in the overload payload. None until first stamped.
_daemon_started_monotonic: float | None = None


# ---------------------------------------------------------------------------
# Config dataclasses + loaders extracted to daemon_config.py
# ---------------------------------------------------------------------------
from iai_mcp.daemon_config import (  # noqa: E402
    ErasureConfig,
    _load_erasure_config,
    PatSepConfig,
    _load_patsep_config,
    S2Config,
    _load_s2_config,
    SleepOverhaulConfig,
    _load_sleep_overhaul_config,
    ReconsolidationConfig,
    _load_reconsolidation_config,
    StcConfig,
    _load_stc_config,
    UserModelConfig,
    _load_user_model_config,
    SpatialConfig,
    _load_spatial_config,
    DmnConfig,
    _load_dmn_config,
    PaskConfig,
    _load_pask_config,
)


async def _hippea_cascade_loop(store, shutdown: asyncio.Event) -> None:
    """Cascade task. Polls `hippea_cascade_request` and pre-warms the
    HIPPEA LRU on pending.

    Invariants:
    - Human-first: yields on shutdown within 5s (via asyncio.wait_for).
    - Zero API cost: no Anthropic SDK import; pure-local salience math.
    - Read-only: cascade is read-only against the store. The ONLY writes
      by this loop are (a) clearing the request flag in state and (b) emitting
      a `hippea_cascade_completed` diagnostic event. Neither mutates
      MemoryRecord rows.

    `retrieve.build_runtime_graph(store)` is wrapped in
    `await asyncio.to_thread(...)` — previously the bare-sync call blocked
    the asyncio event loop for 8-13 s while it traversed the graph. Wrapping
    unblocks every other coroutine on the loop.

    Cascade body is gated by a 60 s minimum-interval cooldown
    (`HIPPEA_CASCADE_MIN_INTERVAL_SEC`). When cooldown blocks, `pending=true`
    STAYS set. Next poll re-checks. Worst-case under perpetual `pending=true`:
    ≤ 1 cascade per 60 s.
    """
    # Explicit `global` so the assignment in the finally block updates
    # module-level state, not a local binding. Without this the cooldown
    # is silently broken.
    global _last_cascade_completed_at

    # Local imports isolate cascade machinery from daemon boot-time cost.
    from iai_mcp import retrieve
    from iai_mcp.daemon_state import load_state, save_state
    from iai_mcp.hippea_cascade import _install_warm, compute_and_fetch_warm

    while not shutdown.is_set():
        try:
            state = await asyncio.to_thread(load_state)
            req = state.get("hippea_cascade_request") or {}
            if req.get("pending"):
                # Cooldown gate: if the cascade body ran within the last
                # MIN_INTERVAL seconds, skip execution but leave pending=True
                # so the next eligible poll runs it.
                elapsed = time.monotonic() - _last_cascade_completed_at
                if elapsed < HIPPEA_CASCADE_MIN_INTERVAL_SEC:
                    # Cooldown active; pending stays set. No event emit
                    # (would flood the ledger every 5s poll).
                    pass
                else:
                    try:
                        assignment = None
                        try:
                            # build_runtime_graph is sync+heavy: off-loop via to_thread.
                            # Returns the 3-tuple (graph, assignment, rich_club).
                            _graph, assignment, _rc = await asyncio.to_thread(
                                retrieve.build_runtime_graph, store,
                            )
                        except (OSError, ValueError, RuntimeError) as exc:
                            log.debug("build_runtime_graph failed in cascade: %s", exc)
                            assignment = None
                        stats: dict = {
                            "communities_selected": 0, "records_warmed": 0,
                            "top_communities": [],
                        }
                        if assignment is not None:
                            try:
                                # Off-loop dispatch: compute_and_fetch_warm runs the
                                # community selection, centrality scan, AND the warm
                                # store.get loop on the dedicated bounded executor
                                # (NOT the default to_thread pool, which the embed
                                # convoy saturates). Only the in-memory LRU insert
                                # (_install_warm) runs back on the loop.
                                loop = asyncio.get_event_loop()
                                executor = _cascade_executor
                                recs, top = await loop.run_in_executor(
                                    executor,
                                    compute_and_fetch_warm,
                                    store,
                                    assignment,
                                )
                                inserted = await _install_warm(recs)
                                stats = {
                                    "communities_selected": len(top),
                                    "records_warmed": inserted,
                                    "top_communities": [str(c) for c in top],
                                }
                            except (OSError, ValueError, RuntimeError) as exc:
                                log.debug("cascade compute+fetch failed: %s", exc)
                                stats = {
                                    "communities_selected": 0,
                                    "records_warmed": 0,
                                    "top_communities": [],
                                }
                        try:
                            await asyncio.to_thread(
                                write_event,
                                store,
                                "hippea_cascade_completed",
                                {
                                    "session_id": req.get("session_id", ""),
                                    **stats,
                                },
                                severity="info",
                            )
                        except (OSError, RuntimeError) as exc:
                            log.debug("hippea_cascade_completed event write failed: %s", exc)
                        # Clear the request flag so we don't re-run the same
                        # cascade. Re-read state before clearing to minimise
                        # lost-write windows with the main tick loop.
                        try:
                            state = await asyncio.to_thread(load_state)
                            state["hippea_cascade_request"] = {"pending": False}
                            await asyncio.to_thread(save_state, state)
                        except (OSError, ValueError) as exc:
                            log.debug("cascade state clear failed: %s", exc)
                    finally:
                        # Stamp end-of-cascade timestamp regardless of
                        # success/exception. Updates module-level state via
                        # the `global` declaration at the top of the function.
                        _last_cascade_completed_at = time.monotonic()
        except Exception:  # noqa: BLE001 -- cascade loop MUST NOT crash
            # Any error in the outer body must not terminate the task
            # (C1: cooperative shutdown only).
            log.warning("hippea cascade loop iteration failed", exc_info=True)
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=HIPPEA_CASCADE_POLL_SEC,
            )
            # shutdown fired -> exit loop
            break
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------
# CPU watchdog body (observation-only)
# ---------------------------------------------------------------------------

def _watchdog_active_task_names() -> list[str]:
    """Best-effort `active_tasks` payload.

    Returns up to 5 names of currently-running asyncio tasks (excluding
    done tasks). Falls back to '?' on empty get_name(). Wrapped in
    try/except so an introspection failure never blocks the event emit.
    """
    out: list[str] = []
    try:
        for t in asyncio.all_tasks():
            if t.done():
                continue
            name = t.get_name() or "?"
            out.append(name)
    except (RuntimeError, AttributeError) as exc:  # noqa: BLE001 -- introspection failure non-fatal
        log.debug("watchdog task introspection failed: %s", exc)
    return out[:5]


async def _cpu_watchdog_loop(store, shutdown: asyncio.Event) -> None:
    """Observation-only CPU watchdog (CPU is NEVER a kill signal).

    Polls own-process CPU every WATCHDOG_POLL_SEC seconds via
    psutil.Process(os.getpid()).cpu_percent(interval=None). When the
    last 2 samples both exceed WATCHDOG_THRESHOLD_PERCENT (default 50),
    emits `daemon_cpu_overload` event with severity=critical containing
    fsm_state, cpu_samples_pct, uptime_sec, active_tasks, threshold_pct,
    sustained_sec.

    Per-event cooldown WATCHDOG_EVENT_COOLDOWN_SEC (default 300s) prevents
    ledger flood under prolonged overload — at most one event per 5 min.

    OBSERVATION-ONLY: no SIGTERM, no os.kill, no launchctl. High CPU means
    the daemon is making consolidation PROGRESS, so it is never killed here;
    the only side-effect is the event emit. Auto-recovery (a controlled
    self-kill -> launchd respawn) lives in the SEPARATE liveness/memory
    watchdog thread, which fires ONLY on a confirmed-unresponsive socket
    round-trip (a wedged loop) or approaching-jetsam memory pressure — never
    on CPU.

    Prime the meter ONCE before the polling loop so the first real sample
    at t=POLL_SEC is a meaningful delta, not a 0.0 baseline-priming response.
    """
    # Explicit `global` so cooldown timestamp updates module
    # state, not a local binding.
    global _last_overload_event_at

    # Local imports: keep daemon boot cheap.
    from collections import deque

    import psutil

    proc = psutil.Process(os.getpid())
    # Prime psutil's internal CPU meter — first cpu_percent
    # call returns 0.0 (no prior measurement to delta against). Discard.
    try:
        proc.cpu_percent(interval=None)
    except (OSError, psutil.Error) as exc:  # noqa: BLE001 -- prime failure non-fatal
        log.debug("psutil cpu_percent prime failed: %s", exc)

    samples: deque[float] = deque(maxlen=WATCHDOG_SAMPLE_WINDOW)

    while not shutdown.is_set():
        # Sleep for one poll interval (or break early on shutdown).
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=WATCHDOG_POLL_SEC,
            )
            break
        except asyncio.TimeoutError:
            pass

        # Sample own-process CPU (delta vs prior call).
        try:
            cpu_pct = proc.cpu_percent(interval=None)
            samples.append(cpu_pct)
        except (OSError, psutil.Error) as exc:  # noqa: BLE001 -- psutil flakiness must not crash
            log.debug("cpu_percent sample failed: %s", exc)
            continue

        # Trigger: 2 consecutive samples both > threshold (= sustained
        # WATCHDOG_POLL_SEC * 2 seconds).
        if (
            len(samples) >= 2
            and samples[-1] > WATCHDOG_THRESHOLD_PERCENT
            and samples[-2] > WATCHDOG_THRESHOLD_PERCENT
        ):
            now_mono = time.monotonic()
            # Cooldown: at most 1 event per 5 min.
            if (now_mono - _last_overload_event_at) < WATCHDOG_EVENT_COOLDOWN_SEC:
                continue

            fsm_state = "?"
            try:
                state = await asyncio.to_thread(load_state)
                fsm_state = state.get("fsm_state", "?")
            except (OSError, ValueError, json.JSONDecodeError) as exc:  # noqa: BLE001 -- introspection only
                log.debug("watchdog load_state failed: %s", exc)

            uptime_sec: float | None = None
            if _daemon_started_monotonic is not None:
                uptime_sec = round(now_mono - _daemon_started_monotonic, 1)

            payload = {
                "fsm_state": fsm_state,
                "cpu_samples_pct": list(samples),
                "uptime_sec": uptime_sec,
                "active_tasks": _watchdog_active_task_names(),
                "threshold_pct": WATCHDOG_THRESHOLD_PERCENT,
                "sustained_sec": int(WATCHDOG_POLL_SEC * 2),
            }

            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "daemon_cpu_overload",
                    payload,
                    severity="critical",
                )
            except (OSError, RuntimeError) as exc:  # noqa: BLE001 -- ledger emit failure non-fatal
                log.debug("daemon_cpu_overload event write failed: %s", exc)
                continue

            _last_overload_event_at = now_mono


# ---------------------------------------------------------------------------
# Liveness + memory self-watchdog: pure decision core (synchronously testable)
# ---------------------------------------------------------------------------
def _next_poll_interval(pressure_level: int | None) -> float:
    """Return the watchdog's next sleep interval given the system memory-pressure
    level.

    NORMAL / unreadable (None or < WARN) -> the steady poll
    (``WATCHDOG_LIVENESS_POLL_SEC``); sustained WARN(>=2) -> the tightened poll
    (``WATCHDOG_WARN_POLL_SEC``). The memory debounce counts TICKS (not
    wall-clock), so tightening the cadence under pressure shortens the reaction
    window (N=3: ~90s at NORMAL -> ~21s at WARN) — a fast RSS balloon keeps its
    debounce window instead of losing it to the jetsam killer.

    Pure: no I/O, no sleep — the thread loop calls it each tick to pick its next
    sleep.
    """
    if pressure_level is not None and pressure_level >= 2:
        return WATCHDOG_WARN_POLL_SEC
    return WATCHDOG_LIVENESS_POLL_SEC


def _evaluate_watchdog(
    probe_ok: bool,
    rss: int | None,
    pressure_level: int | None,
    uptime_sec: float,
    consecutive_failures: int,
    recovery_timestamps: list[float],
    now_wall: float,
    *,
    hard_cap: int,
    contributor_floor: int,
    debounce_n: int,
    cold_start_grace_sec: float,
    max_recoveries: int,
    recovery_window_sec: float,
) -> tuple[str, str]:
    """Decide the watchdog action for one tick. PURE: no I/O, no kill, no sleep.

    Returns ``(action, reason)`` where action is one of:
      - ``"none"``   — healthy / busy / transient / another-process-owns-RAM /
                       within cold-start grace / unreadable pressure.
      - ``"kill"``   — a debounced wedge or memory trigger; reason in
                       {"wedge", "memory", "leak"}.
      - ``"needs_operator"`` — the circuit-breaker tripped (>= max_recoveries
                       self-kills within recovery_window_sec); STOP killing.

    Parameters
    ----------
    probe_ok:
        True iff the active full status round-trip succeeded THIS tick. A wedged
        loop completes ``connect()`` but never replies, so this is computed from
        a full request->reply round-trip (NOT connect-only).
    rss:
        The daemon's own current RSS in bytes (psutil), or None if unreadable.
    pressure_level:
        macOS ``kern.memorystatus_vm_pressure_level`` (1=NORMAL, 2=WARN,
        4=CRITICAL), or None if unreadable. Unreadable => the memory trigger is
        disabled (fail-open — NEVER kill on an unreadable signal); the RSS-leak
        backstop still guards.
    uptime_sec:
        Seconds since boot (monotonic). Within ``cold_start_grace_sec`` the
        memory/leak triggers are suppressed (the boot RSS ramp is legitimate).
        The WEDGE trigger is intentionally NOT grace-covered — a daemon that
        cannot serve a status round-trip N consecutive times after a 5s probe
        timeout is wedged regardless of age (and a launchd re-enable relies
        on the daemon answering status promptly after the socket binds).
    consecutive_failures:
        The running count of consecutive failing ticks INCLUDING this one (the
        caller increments before calling on a triggering tick, resets on a clean
        tick). A trigger acts only when this reaches ``debounce_n``.
    recovery_timestamps:
        Wall-clock epoch timestamps of prior self-kills, reconstructed from the
        on-disk breadcrumb at startup (SIGKILL wipes in-memory state, so the
        circuit-breaker is necessarily cross-process). Counted against the
        window BEFORE deciding to kill.
    now_wall:
        Current wall-clock epoch (``time.time()``). Wall-clock — NOT monotonic —
        because ``recovery_timestamps`` are persisted epochs and monotonic
        resets across a restart.
    """
    # Circuit-breaker FIRST: if we have already self-killed >= max_recoveries
    # times inside the window, a deterministic wedge/leak would otherwise loop
    # SIGKILL->respawn->wedge->SIGKILL forever. Stop killing, surface loudly.
    recent = [t for t in recovery_timestamps if now_wall - t <= recovery_window_sec]
    breaker_tripped = len(recent) >= max_recoveries

    # --- triggers (per-tick observations, no debounce yet) ---
    leak = rss is not None and rss > hard_cap  # backstop: always a leak -> kill
    pressure = pressure_level is not None and pressure_level >= 2  # PRIMARY: WARN
    big = rss is not None and rss > contributor_floor  # daemon is a real contributor
    in_grace = uptime_sec < cold_start_grace_sec

    # Memory/leak triggers are suppressed during cold-start grace (boot RSS ramp).
    mem_trigger = (not in_grace) and (leak or (pressure and big))
    # Wedge trigger is NOT grace-covered (see uptime_sec docstring).
    wedge_trigger = not probe_ok

    if not (mem_trigger or wedge_trigger):
        return ("none", "healthy")

    # Debounce: a trigger acts only when sustained for N consecutive ticks. A
    # single blip never kills.
    if consecutive_failures < debounce_n:
        return ("none", "debounce")

    # Debounce satisfied + a trigger is live. If the breaker has tripped, refuse
    # to kill and surface a loud needs-operator event instead.
    if breaker_tripped:
        return ("needs_operator", "circuit_breaker")

    # Reason precedence: a wedge is the more urgent (cannot-serve) condition; a
    # leak is named distinctly from sustained-pressure memory for the breadcrumb.
    if wedge_trigger:
        return ("kill", "wedge")
    if leak:
        return ("kill", "leak")
    return ("kill", "memory")


# ---------------------------------------------------------------------------
# Liveness + memory self-watchdog: side-effecting helpers (thread + kill path)
# ---------------------------------------------------------------------------
def _watchdog_state_dir() -> "Path":
    """Resolve the iai-mcp state dir for the watchdog breadcrumb log.

    Honors IAI_MCP_STORE (test isolation + multi-tenant) and falls back to
    ~/.iai-mcp in production where the env var is unset.
    """
    root = os.environ.get("IAI_MCP_STORE")
    return Path(root) if root else Path.home() / ".iai-mcp"


def _watchdog_log_path() -> "Path":
    """The append-only plaintext breadcrumb log path."""
    return _watchdog_state_dir() / ".daemon-watchdog.log"


def _watchdog_socket_path() -> str:
    """The control-socket path the probe round-trips against.

    Honors IAI_DAEMON_SOCKET_PATH (the same override the SocketServer + CLI use)
    so tests point at a tmp socket and never the live daemon's socket.
    """
    return os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(
        _watchdog_state_dir() / ".daemon.sock"
    )


def _vm_pressure_level() -> int | None:
    """Read macOS ``kern.memorystatus_vm_pressure_level`` (fork-free, ~µs).

    Returns 1=NORMAL, 2=WARN, 4=CRITICAL, or None when unreadable (non-macOS,
    OID absent, or any libc/ctypes failure). None is fail-open: the memory
    trigger is DISABLED on an unreadable signal — the watchdog NEVER kills on a
    signal it cannot read.
    """
    import ctypes
    import ctypes.util
    import struct

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        size = ctypes.c_size_t(4)
        buf = ctypes.create_string_buffer(4)
        rc = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            buf,
            ctypes.byref(size),
            None,
            0,
        )
        if rc != 0:
            return None
        return struct.unpack("i", buf.raw[:4])[0]
    except Exception:  # noqa: BLE001 -- unreadable pressure must never crash/kill
        return None


def _own_rss_bytes() -> int | None:
    """The daemon's own current RSS in bytes (psutil), or None if unreadable.

    Current RSS (memory_info().rss), NOT ru_maxrss — the trigger watches the
    live footprint, not a historical peak.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:  # noqa: BLE001 -- psutil flakiness must not crash/kill
        return None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_breadcrumb(line: bytes) -> None:
    """Best-effort raw os.write to the pre-opened breadcrumb fd.

    LOCK-FREE: a raw ``os.write`` to an already-open fd — NO write_event, NO
    HippoTable.add, NO connection lock. Any failure (fd closed/invalid, disk
    full, fd unset) is swallowed; the breadcrumb is best-effort ONLY and can
    NEVER prevent the kill. Isolated as a one-liner so a test can force it to
    raise (an invalid fd -> EBADF) without globally monkeypatching os.write.
    """
    fd = _WATCHDOG_LOG_FD
    if fd is None:
        raise OSError("watchdog breadcrumb fd not open")
    os.write(fd, line)


def _self_kill(reason: str, kind: str) -> None:
    """Lock-free, UNCONDITIONAL controlled self-kill — shared by BOTH triggers.

    Writes ONE best-effort lock-free breadcrumb line (so a post-mortem and the
    cross-process circuit-breaker can read it), then SIGKILLs THIS process
    UNCONDITIONALLY. The kill is NEVER inside the try and is NEVER gated on the
    breadcrumb succeeding: the worst case the watchdog exists to avert —
    jetsam during consolidation — is precisely when the consolidation worker
    holds the store connection lock, so a kill that first took that lock (via
    write_event) would block forever and the SIGKILL would never run. ONE helper
    covers the wedge AND the memory path, structurally guaranteeing both are
    lock-independent and cannot diverge.

    ``reason`` ∈ {"wedge","memory","leak"}; ``kind`` is the event-kind token
    (DAEMON_WEDGE_KILL / DAEMON_MEMORY_PRESSURE_KILL), carried in the breadcrumb.
    The breadcrumb line is also what reconstructs the circuit-breaker across the
    SIGKILL (in-memory recovery state is wiped on every respawn).
    """
    try:
        line = f"{_iso_now()} {kind} reason={reason} pid={os.getpid()}\n".encode()
        _write_breadcrumb(line)
    except Exception:  # noqa: BLE001 -- breadcrumb is best-effort ONLY
        pass
    os.kill(os.getpid(), signal.SIGKILL)  # UNCONDITIONAL — never gated on the emit


def _capture_blackbox(
    log_fd: int | None,
    probe_ok: bool,
    consecutive_failures: int,
    debounce_n: int,
) -> None:
    """Lock-free forensic dump on a failing probe tick (below the kill threshold).

    Writes a structured header then calls ``faulthandler.dump_traceback`` (all
    threads) to the pre-opened fd. Also captures: FD count (via /dev/fd listing),
    running asyncio task names (best-effort introspection), and pid/timestamp.

    LOCK-FREE contract: uses only ``os.write`` to a pre-opened fd + ``faulthandler``.
    No SQLite, no ``_conn_lock``, no ``write_event``, no allocation beyond a small
    header string. Safe to call when the event loop thread may be stalled and any
    store connection lock may be held.

    LOGS ONLY — never calls ``_self_kill``, never modifies the kill-decision state.
    If the fd is None or any step fails, the failure is silently swallowed so the
    watchdog tick is never interrupted by a capture error.
    """
    if log_fd is None:
        return
    try:
        # FD count: best-effort, fail-open (None if the /dev/fd directory
        # is not readable on this platform).
        try:
            fd_count: int | None = len(os.listdir("/dev/fd"))
        except OSError:
            fd_count = None

        # Running asyncio task names: safe from a non-loop thread; may return
        # an empty list if no loop is running — that is fine.
        task_names: list[str] = []
        try:
            task_names = _watchdog_active_task_names()
        except Exception:  # noqa: BLE001
            pass

        # Write a structured single-line header.
        header = (
            f"{_iso_now()} pre_kill_forensic_dump"
            f" pid={os.getpid()}"
            f" probe_ok={probe_ok}"
            f" consecutive_failures={consecutive_failures}"
            f" debounce_n={debounce_n}"
            f" fd_count={fd_count}"
            f" tasks={task_names}\n"
        ).encode()
        try:
            os.write(log_fd, header)
        except OSError:
            pass  # best-effort only

        # Full all-thread traceback — the load-bearing payload.
        try:
            faulthandler.dump_traceback(log_fd, all_threads=True)
        except Exception:  # noqa: BLE001 -- faulthandler failure is non-fatal
            pass

        # Trailing separator so successive dumps are easy to grep.
        try:
            os.write(log_fd, b"--- end dump ---\n")
        except OSError:
            pass
    except Exception:  # noqa: BLE001 -- capture failure must never crash the watchdog
        pass


async def _open_exclusive_store_with_backoff(
    store_factory,
    *,
    max_attempts: int | None = None,
    backoff_sec: float | None = None,
):
    """Await a bounded retry/backoff on ``HippoLockHeldError`` from the EXCLUSIVE
    store open at boot.

    When a prior daemon instance is killed mid-consolidation, the process-level
    ``fcntl.LOCK_EX`` on the Hippo ``.lock`` file may not yet have been released
    by the dying process when the new instance starts. A bare ``MemoryStore(EXCLUSIVE)``
    construction raises ``HippoLockHeldError`` in that race. This helper retries
    up to ``max_attempts`` times with an increasing ``asyncio.sleep`` delay so the
    respawn waits for the predecessor to fully exit rather than crash-looping.

    Contract:
    - Each attempt calls ``store_factory()`` synchronously (the construction is sync).
    - Only ``HippoLockHeldError`` is retried; any other exception propagates immediately.
    - After ``max_attempts`` failed attempts the final ``HippoLockHeldError`` is
      re-raised, surfacing loud and preserving the single-owner guarantee.
    - Uses ``await asyncio.sleep`` (not ``time.sleep``) so the main event loop stays
      responsive during the boot wait.
    """
    from iai_mcp.hippo import HippoLockHeldError as _HippoLockHeldError  # local import to avoid circular

    _max = max_attempts if max_attempts is not None else BOOT_LOCK_RETRY_ATTEMPTS
    _base = backoff_sec if backoff_sec is not None else BOOT_LOCK_RETRY_BACKOFF_SEC

    last_exc: _HippoLockHeldError | None = None
    for attempt in range(1, _max + 1):
        try:
            return store_factory()
        except _HippoLockHeldError as exc:
            last_exc = exc
            if attempt < _max:
                delay = _base * attempt  # linear back-off: 0.5s, 1.0s, 1.5s, ...
                log.warning(
                    "exclusive store open: lock held by predecessor "
                    "(attempt %d/%d) — retrying in %.2f s",
                    attempt,
                    _max,
                    delay,
                )
                await asyncio.sleep(delay)
    # Exhausted — re-raise so the caller surfaces the failure loud.
    assert last_exc is not None  # guaranteed: loop ran at least once with failure
    log.error(
        "exclusive store open: lock still held after %d attempts — giving up",
        _max,
    )
    raise last_exc


def _load_recovery_timestamps(
    log_path: "Path", kinds: tuple[str, ...]
) -> list[float]:
    """Reconstruct prior self-kill wall-clock timestamps from the breadcrumb log.

    SIGKILL is uncatchable and wipes all in-memory state on every respawn, so an
    in-memory recovery list would reset to empty each boot and the circuit-
    breaker would NEVER trip — SIGKILL->respawn->wedge->SIGKILL forever. The
    breadcrumb log is the lock-free cross-process sink: each kill line carries an
    ISO wall-clock stamp + its kind. Parse the kill lines, return their epochs
    (best-effort — a malformed/partial line is skipped; under-counting slightly
    is acceptable, NO persistence would be the disaster).

    WALL-CLOCK epochs (time.time()): they are compared against time.time() at
    decision time, NOT monotonic (which resets across a restart).
    """
    out: list[float] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                parts = raw.split(None, 2)
                if len(parts) < 2 or parts[1] not in kinds:
                    continue
                try:
                    dt = datetime.fromisoformat(parts[0])
                    out.append(dt.timestamp())
                except (ValueError, OverflowError):
                    continue
    except FileNotFoundError:
        return []
    except OSError:
        return []
    return out


async def _probe_status_roundtrip(sock_path: str, read_timeout: float) -> bool:
    """Active FULL status round-trip probe (request -> reply), NOT connect-only.

    A wedged event loop still completes ``connect()`` (the kernel accept queue),
    so connect-only is the WRONG primitive — only a full request->reply proves
    the loop SERVES. Returns True iff a non-empty reply line came back inside
    ``read_timeout``; False on any connect failure, a missing socket, an empty
    reply, or a read that exceeds the timeout (the wedge signal).

    Runs via ``asyncio.run`` in the watchdog THREAD's own throwaway loop — NEVER
    the daemon's loop (the whole point is to survive a wedged daemon loop).
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(sock_path), timeout=5.0
        )
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    except asyncio.TimeoutError:
        return False
    try:
        writer.write((json.dumps({"type": "status"}) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=read_timeout)
        return bool(line)
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        try:
            writer.close()
        except OSError:
            pass


def _watchdog_tick(
    store,
    sock_path: str,
    log_path: "Path",
    consecutive_failures: int,
    *,
    probe_fn=None,
    pressure_fn=None,
    rss_fn=None,
    blackbox_fn=None,
) -> tuple[float, int]:
    """Run ONE watchdog tick synchronously and act. Returns (next_interval,
    consecutive_failures).

    Isolated from the thread loop so it is synchronously testable: no real
    thread, no real sleep, no real socket required (the probe/pressure/rss
    sources are injectable). It gathers the inputs, calls the pure
    ``_evaluate_watchdog``, and acts:
      - action == "kill"          -> ``_self_kill`` (lock-free breadcrumb then
                                     UNCONDITIONAL SIGKILL). The breadcrumb line
                                     IS the persisted recovery timestamp the next
                                     respawn's circuit-breaker reads.
      - action == "needs_operator"-> emit the loud needs-operator event via
                                     write_event (NO kill, so a blocked emit only
                                     delays the loud event — acceptable).
      - action == "none"          -> reset/advance the debounce counter.

    ``blackbox_fn`` is the forensic-dump hook, injectable for tests (signature:
    ``fn(log_fd, probe_ok, consecutive_failures, debounce_n) -> None``). When
    ``None`` the real ``_capture_blackbox`` is used if enabled. The capture fires
    ONCE per failure episode (on the first failing tick) BEFORE the kill decision,
    so the dump exists even when a kill follows. It never calls ``_self_kill`` and
    never alters the tick decision.
    """
    global _WATCHDOG_BLACKBOX_EPISODE_FIRED

    probe_fn = probe_fn or _probe_status_roundtrip
    pressure_fn = pressure_fn or _vm_pressure_level
    rss_fn = rss_fn or _own_rss_bytes

    # 1. Active full status round-trip (in a throwaway loop — survives a wedge).
    try:
        probe_ok = asyncio.run(
            probe_fn(sock_path, WATCHDOG_PROBE_TIMEOUT_SEC)
        )
    except Exception:  # noqa: BLE001 -- a probe failure counts as not-ok, never crashes
        probe_ok = False

    # 2. System memory-pressure + own RSS (both fail-open to None).
    pressure_level = pressure_fn()
    rss = rss_fn()

    # 3. Debounce bookkeeping: a tick that observes ANY live trigger advances
    #    the counter; a clean tick resets it. We recompute the per-tick trigger
    #    cheaply here (same predicates as _evaluate_watchdog) to drive debounce.
    leak = rss is not None and rss > WATCHDOG_RSS_HARD_CAP_BYTES
    pressure = pressure_level is not None and pressure_level >= 2
    big = rss is not None and rss > WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES
    uptime_sec = (
        time.monotonic() - _daemon_started_monotonic
        if _daemon_started_monotonic is not None
        else 1e9  # treat unknown-uptime as past grace (fail toward acting)
    )
    in_grace = uptime_sec < WATCHDOG_COLD_START_GRACE_SEC
    mem_trigger = (not in_grace) and (leak or (pressure and big))
    tick_failing = (not probe_ok) or mem_trigger
    consecutive_failures = consecutive_failures + 1 if tick_failing else 0

    # 3b. Forensic pre-kill black box: fires ONCE per failure episode on a
    #     failing probe tick BEFORE the kill threshold.
    #     Criteria: probe failed AND below the kill threshold AND not yet fired
    #     this episode. A clean tick resets the episode gate.
    if not tick_failing:
        # Clean tick: reset the episode flag so the next episode dumps again.
        _WATCHDOG_BLACKBOX_EPISODE_FIRED = False
    elif (
        not probe_ok
        and consecutive_failures < WATCHDOG_FAILURE_DEBOUNCE_N
        and not _WATCHDOG_BLACKBOX_EPISODE_FIRED
    ):
        _WATCHDOG_BLACKBOX_EPISODE_FIRED = True
        # Resolve the capture fn: use the injected one (tests) or the real one.
        _bb_fn = blackbox_fn
        if _bb_fn is None and _WATCHDOG_BLACKBOX_ENABLED:
            _bb_fn = _capture_blackbox
        if _bb_fn is not None:
            try:
                _bb_fn(
                    _WATCHDOG_BLACKBOX_FD,
                    probe_ok,
                    consecutive_failures,
                    WATCHDOG_FAILURE_DEBOUNCE_N,
                )
            except Exception:  # noqa: BLE001 -- capture failure must never interrupt the watchdog
                pass

    # 4. Cross-process circuit-breaker state (reconstructed from disk).
    recovery_timestamps = _load_recovery_timestamps(
        log_path, (DAEMON_WEDGE_KILL, DAEMON_MEMORY_PRESSURE_KILL)
    )

    # 5. Pure decision.
    action, reason = _evaluate_watchdog(
        probe_ok,
        rss,
        pressure_level,
        uptime_sec,
        consecutive_failures,
        recovery_timestamps,
        time.time(),  # WALL-CLOCK for the recovery window (matches persisted epochs)
        hard_cap=WATCHDOG_RSS_HARD_CAP_BYTES,
        contributor_floor=WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES,
        debounce_n=WATCHDOG_FAILURE_DEBOUNCE_N,
        cold_start_grace_sec=WATCHDOG_COLD_START_GRACE_SEC,
        max_recoveries=WATCHDOG_MAX_RECOVERIES,
        recovery_window_sec=WATCHDOG_RECOVERY_WINDOW_SEC,
    )

    # 6. Act.
    if action == "kill":
        kind = (
            DAEMON_WEDGE_KILL if reason == "wedge" else DAEMON_MEMORY_PRESSURE_KILL
        )
        # The breadcrumb (written inside _self_kill) is the persisted recovery
        # timestamp the next respawn's circuit-breaker reads — no separate write.
        _self_kill(reason, kind)
        # _self_kill SIGKILLs unconditionally; control never returns here in
        # production. In tests os.kill is mocked, so fall through cleanly.
    elif action == "needs_operator":
        # No kill -> write_event is acceptable (a blocked emit only delays the
        # loud event; it is not a safety hazard).
        try:
            write_event(
                store,
                DAEMON_WATCHDOG_NEEDS_OPERATOR,
                {
                    "reason": reason,
                    "consecutive_failures": consecutive_failures,
                    "recoveries_in_window": len(recovery_timestamps),
                    "max_recoveries": WATCHDOG_MAX_RECOVERIES,
                },
                severity="critical",
            )
        except Exception:  # noqa: BLE001 -- a loud-event emit failure is non-fatal
            log.debug("watchdog needs_operator emit failed", exc_info=True)

    return (_next_poll_interval(pressure_level), consecutive_failures)


def _liveness_watchdog(store, stop_event, sock_path: str | None = None) -> None:
    """Self-watchdog THREAD body (NOT an asyncio task — must survive a wedged
    loop).

    Opens the lock-free breadcrumb fd and the forensic black-box dump fd ONCE
    at start (so the kill path needs no allocation), then loops: run one
    ``_watchdog_tick`` (active probe + memory check + pure decision + act), sleep
    the adaptive interval (steady at NORMAL, tightened at WARN), breakable on
    ``stop_event`` (a threading.Event — a real thread cannot await an asyncio.Event).
    """
    global _WATCHDOG_LOG_FD, _WATCHDOG_BLACKBOX_FD, _WATCHDOG_BLACKBOX_EPISODE_FIRED

    if sock_path is None:
        sock_path = _watchdog_socket_path()
    log_path = _watchdog_log_path()

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _WATCHDOG_LOG_FD = os.open(
            str(log_path),
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
    except OSError:
        # Without the breadcrumb fd the kill is still UNCONDITIONAL (the
        # breadcrumb is best-effort), but the cross-process circuit-breaker
        # cannot reconstruct prior kills, so a deterministic wedge could loop.
        # Log and continue — launchd's ThrottleInterval is the outer backstop.
        log.warning("watchdog breadcrumb fd open failed; circuit-breaker degraded")
        _WATCHDOG_LOG_FD = None

    # Open the forensic black-box dump fd (separate from the breadcrumb log so
    # multi-line faulthandler output never corrupts the circuit-breaker parser).
    if _WATCHDOG_BLACKBOX_ENABLED:
        try:
            bb_log_path = log_path.with_name(".daemon-blackbox.log")
            _WATCHDOG_BLACKBOX_FD = os.open(
                str(bb_log_path),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
        except OSError:
            log.debug("watchdog black-box fd open failed; forensic dump disabled")
            _WATCHDOG_BLACKBOX_FD = None

    # Reset the episode flag on (re-)start so any prior run's state is cleared.
    _WATCHDOG_BLACKBOX_EPISODE_FIRED = False

    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            next_interval, consecutive_failures = _watchdog_tick(
                store, sock_path, log_path, consecutive_failures
            )
        except Exception:  # noqa: BLE001 -- the watchdog must NEVER crash the daemon
            log.debug("watchdog tick failed", exc_info=True)
            next_interval = WATCHDOG_LIVENESS_POLL_SEC
        # Breakable sleep on the threading.Event — wakes immediately on shutdown.
        stop_event.wait(timeout=next_interval)


# ---------------------------------------------------------------------------
# The RSS watchdog + clean-shutdown restart trigger block have been removed.
# `_resolve_shutdown_exit_code` (75/0 sentinel decision),
# `_clean_shutdown_for_restart` (os._exit(75)), and `_rss_watchdog_loop`
# (RSS polling + TTL trigger) are all gone.
#
# The lifecycle state machine + sleep_pipeline supersede this design.
# Hibernation kills the process with exit 0 (graceful) and the plist's
# `KeepAlive={"Crashed": true}` ensures launchd does NOT auto-respawn
# on graceful exit; the wrapper kickstart is the wake mechanism.
#
# The user-stop sentinel is PRESERVED but simplified.
# `iai-mcp daemon stop` still writes `user_requested_shutdown=True`
# to `.daemon-state.json` before SIGTERM; the daemon's main() finally
# block clears the sentinel from the on-disk file (so a stale flag
# cannot leak across boots) but the exit code is now uniformly 0
# regardless of who triggered the shutdown.
# ---------------------------------------------------------------------------

# Sentinel key in .daemon-state.json. The daemon's main() finally block
# clears the on-disk flag so it does not leak across boots; the exit code
# does not branch on it (always 0).
_USER_SHUTDOWN_FLAG = "user_requested_shutdown"


def _clear_user_shutdown_sentinel(state: dict) -> None:
    """Clear the on-disk + in-memory ``user_requested_shutdown`` flag.

    Cross-process invariant (preserved from 541c874): the CLI
    ``iai-mcp daemon stop`` runs in a SEPARATE process from the daemon
    and writes the sentinel to ``.daemon-state.json`` BEFORE sending
    SIGTERM. The daemon's in-memory ``state`` dict was loaded at boot
    time and is never re-read on signal — so the disk-side flag must
    be cleared explicitly here, not just popped from the memory dict.

    The function ONLY clears the sentinel; it does NOT decide an exit code.
    main() always returns 0 on graceful shutdown, regardless of who triggered
    it. launchd's ``KeepAlive={"Crashed": true}`` plist ensures graceful exit 0
    stays dead until wrapper kickstart fires.

    Read failure is fail-safe: ignored. The next ``save_state`` from
    main() will overwrite the on-disk record anyway.
    """
    try:
        on_disk = load_state()
        if _USER_SHUTDOWN_FLAG in on_disk:
            on_disk.pop(_USER_SHUTDOWN_FLAG, None)
            save_state(on_disk)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # Disk read/write failure must NOT block shutdown.
        log.debug("clear_user_shutdown_sentinel disk op failed: %s", exc)
    state.pop(_USER_SHUTDOWN_FLAG, None)


# ---------------------------------------------------------------------------
# Held warm-embedder singleton (long-lived-process serving optimization)
# ---------------------------------------------------------------------------

def _install_warm_embedder_override(store) -> tuple[object, bool]:
    """Build ONE warm embedder via the captured funnel and HOLD it.

    The long-lived process serves concurrent in-process recalls; it builds
    a single warm embedder at boot and installs a process-local override of
    the ``embedder_for_store`` funnel so every reuse site returns that same
    held instance with zero per-call reconstruction. The instance is built
    THROUGH the captured funnel (``orig(store)``) — never a direct
    ``Embedder()`` construct — so the funnel's dim path and any test stub
    are honored, and the held copy is the same single English model.

    Returns ``(orig_efs, installed)``:
      - ``orig_efs`` is the original funnel, to be restored on shutdown.
      - ``installed`` is True only if the override was actually set.

    Build/hold failure is NON-FATAL: on any exception the default funnel is
    left in place (``installed`` False), so reuse sites construct fresh
    (status quo, safe) and the process still boots. The held instance lives
    for the process's serving lifetime; it dies with the process on exit
    and is re-held on the next start.
    """
    import iai_mcp.embed as _embed_mod

    orig_efs = _embed_mod.embedder_for_store
    try:
        warm = orig_efs(store)
        warm.embed("warmup")  # page-cache touch + smoke encode

        def _held_embedder_for_store(_store):
            return warm

        _embed_mod.embedder_for_store = _held_embedder_for_store
        return orig_efs, True
    except Exception as exc:  # noqa: BLE001 -- prewarm/hold failure is non-fatal
        log.warning("embedder prewarm/hold failed: %s", exc, exc_info=True)
        try:
            write_event(store, "prewarm_failed", {"error": str(exc)}, severity="warning")
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("prewarm_failed event write failed")
        # Leave the default funnel installed -> reuse sites construct fresh.
        return orig_efs, False


def _restore_embedder_funnel(orig_efs: object, installed: bool) -> None:
    """Restore the original ``embedder_for_store`` funnel on shutdown.

    Guarded by ``installed``: a build/hold failure (override never set) or an
    early lock-conflict return (install never reached) must NOT restore a
    never-captured value. Shutdown must never crash on restore.
    """
    if not installed:
        return
    try:
        import iai_mcp.embed as _embed_mod

        _embed_mod.embedder_for_store = orig_efs
    except Exception:  # noqa: BLE001 -- shutdown must never crash on restore
        log.debug("embedder funnel restore failed", exc_info=True)


# ---------------------------------------------------------------------------
# Process title
# ---------------------------------------------------------------------------

def _set_process_title(title: str = "iai lilli (iai_mcp.daemon)") -> None:
    """Set the OS-level process title. Reads as the brand "iai lilli" in
    Activity Monitor while keeping the "iai_mcp.daemon" token in the
    command line so process-identification by cmdline substring (the
    lockfile liveness cross-check, ``daemon stop`` PID confirmation, doctor)
    keeps recognising the daemon. Fail-soft: an absent or broken
    setproctitle must never block daemon boot."""
    try:
        from setproctitle import setproctitle as _setproctitle
        _setproctitle(title)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    """Open store, prewarm embedder, serve socket, tick forever.

    Returns 0 on clean shutdown (signal-driven OR Hibernation transition);
    returns 1 only on LifecycleLockConflict (a same-host live-PID conflict);
    raises SystemExit(2) on partial-migration boot block. Signals
    SIGTERM/SIGINT/SIGHUP all set the shutdown event.

    Tasks spawned:
    - mcp_socket_task:       SocketServer.serve() — SOLE binder of the daemon
                             control socket.
    - tick_task:             scheduler tick loop (_scheduler_tick + _tick_body)
                             for consolidation REM cycles. The in-process yield
                             gate inside _tick_body was removed; the lifecycle
                             state machine supersedes it.
    - audit_task:            continuous_audit (read-only).
    - s4_task:               hourly S4 offline pass.
    - cascade_task:          activation-cascade pre-warmer.
    - cpu_watchdog_task:     observation-only CPU watchdog.
    - lifecycle_tick_task:   drives the WAKE/DROWSY/SLEEP/HIBERNATION state
                             machine every 30 s; runs the sleep pipeline on
                             SLEEP entry; sets the global shutdown event on
                             HIBERNATION (with shadow_run=False).
    """
    _set_process_title()
    _require_native()
    _raise_fd_limit()

    # Open in EXCLUSIVE mode for the boot integrity rebuild.
    # After the first WAKE transition the lock is downgraded to SHARED so
    # concurrent SHARED clients can open the store while the daemon is awake.
    #
    # A prior daemon that was SIGKILL'd mid-consolidation may not yet have
    # released its fcntl LOCK_EX on hippo/.lock when this respawn starts.
    # _open_exclusive_store_with_backoff retries with bounded backoff so the
    # respawn waits for the predecessor to exit rather than crash-looping on
    # HippoLockHeldError. The backoff is bounded and fail-loud on exhaustion,
    # preserving the single-owner EXCLUSIVE guarantee.
    store = await _open_exclusive_store_with_backoff(
        lambda: MemoryStore(
            read_consistency_interval=timedelta(seconds=0),
            access_mode=AccessMode.EXCLUSIVE,
        )
    )

    # Clean up any stale consolidation-intent flag left by a prior crashed
    # daemon so clients are not permanently blocked on first boot.
    try:
        hippo_lock_path = store.root / "hippo" / ".lock"
        cleanup_stale_consolidation_intent(hippo_lock_path)
    except Exception:  # noqa: BLE001
        pass

    try:
        from iai_mcp.crypto_key_watch import check_crypto_key_file_rotation_event

        check_crypto_key_file_rotation_event(store)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        log.debug("crypto key rotation check skipped: %s", exc)

    # Boot-time partial-migration detector. The rollback handler in migrate.py
    # only fires if it's actually called from the boot path. Placed BEFORE the
    # embedder prewarm so a partial-state boot short-circuits before paying
    # the ~10s model-load cost.
    #
    # State machine (see migrate.detect_partial_migration):
    #   - clean / unknown          -> proceed to ready advertisement.
    #   - needs_cleanup            -> drop records_old_<ts>, then proceed.
    #   - needs_rollback           -> STOP daemon; surface remediation prompt.
    #   - partial_swap_inconsistent -> STOP daemon; surface remediation prompt
    #                                  (manual recovery; no rollback anchor).
    from iai_mcp.migrate import detect_partial_migration
    _migration_state = detect_partial_migration(store.db)
    if _migration_state["state"] == "partial_swap_inconsistent":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback to restore from "
                        "records_old_<ts>, then iai-mcp daemon-start."
                    ),
                }) + "\n"
            )
        except (OSError, ValueError, TypeError) as exc:
            log.debug("stderr write for partial_swap_inconsistent failed: %s", exc)
        raise SystemExit(2)
    if _migration_state["state"] == "needs_rollback":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback (discard the partial "
                        "staging) OR iai-mcp migrate --resume (continue "
                        "from migration_progress.json checkpoint)."
                    ),
                }) + "\n"
            )
        except (OSError, ValueError, TypeError) as exc:
            log.debug("stderr write for needs_rollback failed: %s", exc)
        raise SystemExit(2)
    if _migration_state["state"] == "needs_cleanup":
        # Successful swap from a previous boot; drop the old table now.
        for _old_name in _migration_state.get("old_tables", []):
            try:
                store.db.drop_table(_old_name)
            except (OSError, RuntimeError, KeyError) as _exc:
                log.warning("migrate cleanup drop_table(%s) failed: %s", _old_name, _exc)
                try:
                    sys.stderr.write(
                        json.dumps({
                            "event": "migrate_cleanup_failed",
                            "table": _old_name,
                            "err": str(_exc)[:120],
                        }) + "\n"
                    )
                except (OSError, ValueError, TypeError):
                    pass

    # Boot-time validation: fail loud now rather than mid sleep-cycle.
    # All loaders are CALL-ON-DEMAND so per-step handlers and constructors
    # can re-invoke fresh inside their bodies for test-monkeypatch support.
    # Return values are discarded; calls exist solely to surface ValueError
    # before the embedder prewarm pays its ~10s cold-start cost.
    _load_erasure_config()
    _load_patsep_config()
    _load_s2_config()
    _load_sleep_overhaul_config()
    _load_reconsolidation_config()
    # Validates 4 IAI_MCP_* env vars (PERI_EVENT_BUFFER_SIZE,
    # PERI_EVENT_WINDOW_SEC, STC_STRONG_EVENT_TYPES, STC_DRY_RUN).
    _load_stc_config()
    # Validates 3 IAI_MCP_* env vars (DMN_REFLECTION_WINDOW_HOURS,
    # META_ANALYST_ENABLED, DMN_DRY_RUN).
    _load_dmn_config()
    # Validates 2 IAI_MCP_PASK_* env vars (ENABLED, DRY_RUN).
    _load_pask_config()

    # Held warm-embedder bookkeeping. The HOLD + funnel override is installed
    # AFTER the lifecycle-lock gate (below) so an early lock-conflict return
    # never installs/leaks the override; the shutdown finally restores the
    # original funnel guarded by `_override_installed`. Initialized here so the
    # finally can never hit an UnboundLocalError on any early exit.
    _orig_efs: object = None
    _override_installed = False

    # Acquire the single-machine lifecycle lockfile (the `.locked` PID marker).
    # This is the higher-level, human-readable singleton marker for the
    # lifecycle state machine. A live-PID conflict on the same host raises
    # LifecycleLockConflict and we exit 1; dead-PID or foreign-host scenarios
    # are silently overwritten.
    from iai_mcp.lifecycle_lock import LifecycleLock, LifecycleLockConflict

    lifecycle_lock = LifecycleLock()
    try:
        lifecycle_lock.acquire()
    except LifecycleLockConflict as exc:
        sys.stderr.write(f"daemon already running: {exc}\n")
        return 1

    # Hold ONE warm embedder for this process's serving lifetime. Installed
    # AFTER the lock gate so the early lock-conflict `return 1` above never
    # leaks the override. Build/hold failure is non-fatal (default funnel
    # stays, reuse sites construct fresh). The held instance dies with the
    # process on shutdown/hibernation and is re-held on the next start; the
    # shutdown finally restores the original funnel (guarded by the sentinel).
    _orig_efs, _override_installed = _install_warm_embedder_override(store)

    # Everything after the warm-embedder override install runs inside this
    # try whose finally restores the original funnel on ANY exit (clean
    # shutdown, hibernation-transition return, or a raised boot exception).
    # The restore touches only the pre-initialized sentinel pair, so the
    # finally can never UnboundLocalError on an early boot raise.
    try:
        # Detect drift between the canonical and legacy state files. Detect-only:
        # the canonical lifecycle_state.json is the source of truth. A mismatch
        # is surfaced via a warning-severity event and a stdlib log line so an
        # operator can audit; the daemon does not auto-correct.
        try:
            from iai_mcp.fsm_reconcile import reconcile_fsm_state

            _drift_report = reconcile_fsm_state(auto_correct=True)
            if _drift_report.get("drift") is True:
                log.warning(
                    "fsm_drift_detected canonical=%s legacy=%s",
                    _drift_report.get("canonical"),
                    _drift_report.get("legacy"),
                )
                try:
                    write_event(
                        store,
                        "fsm_drift_detected",
                        _drift_report,
                        severity="warning",
                        domain="ops",
                    )
                except Exception:  # noqa: BLE001 -- fail-safe
                    log.debug("fsm_drift_detected event write failed")
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            log.debug("fsm_reconcile failed", exc_info=True)

        # Archive any HIBERNATION-stuck.bak recovery artifacts left by prior
        # daemon lives. Pure disk hygiene; fail-safe.
        try:
            from iai_mcp.archive_backups import archive_stuck_backups

            archive_stuck_backups()
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            log.debug("archive_stuck_backups failed", exc_info=True)

        state = await asyncio.to_thread(load_state)
        state.setdefault("fsm_state", STATE_WAKE)
        state["daemon_started_at"] = datetime.now(timezone.utc).isoformat()
        # Stamp monotonic boot time so CPU watchdog payload can include
        # uptime_sec. Module-level global; written here only.
        global _daemon_started_monotonic
        _daemon_started_monotonic = time.monotonic()
        # Stamp daemon_pid into the state file so `iai-mcp doctor` check (a)
        # can read the live PID. The fcntl `.lock` file holds zero PID bytes,
        # so a separate source of truth is required. On graceful shutdown the
        # finally block clears this key (see below).
        state["daemon_pid"] = os.getpid()
        await asyncio.to_thread(save_state, state)
        write_event(store, "daemon_started", {"state": state["fsm_state"]})

        # Consume any pending wake.signal written by the MCP wrapper while
        # the daemon was down. A consumed wake_signal dispatches WAKE_SIGNAL
        # to the LSM (transitions HIBERNATION -> WAKE if needed; no-op on
        # cold boot where current_state is already WAKE).
        _wake_was_pending = False
        try:
            from pathlib import Path as _Path

            from iai_mcp.wake_handler import WakeHandler

            _wake_signal_path = _Path("~/.iai-mcp/wake.signal").expanduser()
            if WakeHandler(_wake_signal_path).consume_wake_signal():
                _wake_was_pending = True
                write_event(
                    store, "wake_signal_consumed", {"phase": "startup"}, severity="info"
                )
        except Exception:  # noqa: BLE001 -- boot MUST NOT block on wake-handler
            # Defensive: never block daemon boot on a wake-handler error.
            log.debug("wake signal consume failed", exc_info=True)

        # Drain any capture-queue records buffered by the wrapper while the
        # daemon was hibernated. Records are routed back through the existing
        # capture path so the verbatim contract is preserved end-to-end.
        try:
            from iai_mcp.capture import capture_turn as _capture_turn
            from iai_mcp.capture_queue import CaptureQueue

            _capture_queue = CaptureQueue()
            # Bind store via closure; map the queue's record envelope to
            # capture_turn's keyword-only signature (cue, text, tier,
            # session_id, role). The queue's records originate from the
            # wrapper's memory_capture path which already populates these
            # fields verbatim.
            def _capture_handler(record: dict) -> None:
                kwargs = {
                    "cue": record.get("cue", ""),
                    "text": record.get("text", record.get("surface", "")),
                    "tier": record.get("tier", "episodic"),
                    "session_id": record.get("session_id", "-"),
                    "role": record.get("role", "user"),
                }
                _capture_turn(store, **kwargs)

            ingested = await asyncio.to_thread(
                _capture_queue.ingest_pending, _capture_handler,
            )
            if ingested > 0:
                write_event(
                    store,
                    "capture_queue_drained",
                    {"phase": "startup", "ingested": ingested},
                    severity="info",
                )
        except Exception as exc:  # noqa: BLE001 -- never block boot on queue drain
            log.warning("capture queue drain failed at startup: %s", exc, exc_info=True)
            try:
                write_event(
                    store,
                    "capture_queue_drain_failed",
                    {"phase": "startup", "error": str(exc)[:200]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("capture_queue_drain_failed event write failed")

        # Startup-prune: drain any first_turn_pending entries that are older
        # than FIRST_TURN_PENDING_TTL_SEC_DEFAULT (1h). Stale entries
        # perpetually retrigger the activation cascade. Pruning at boot resets
        # the slate; the per-tick prune (in
        # _tick_body Step 0.5) keeps it clean during long-running daemons.
        #
        # We pass an explicit `now=` kwarg (rather than letting the helper
        # default to `datetime.now(timezone.utc)`) so the helper's behaviour
        # is fully deterministic from the caller's perspective. Tests of the
        # wire-in can supply a fixed `NOW` and assert the helper output
        # directly without datetime monkeypatching.
        try:
            from iai_mcp.daemon_state import (
                FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                prune_first_turn_pending,
            )

            state, dropped = prune_first_turn_pending(
                state, now=datetime.now(timezone.utc),
            )
            if dropped:
                await asyncio.to_thread(save_state, state)
                try:
                    write_event(
                        store,
                        "first_turn_pending_expired",
                        {
                            "dropped_count": len(dropped),
                            "session_ids": dropped,
                            "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                            "phase": "startup",
                        },
                        severity="info",
                    )
                except (OSError, RuntimeError) as exc:
                    log.debug("first_turn_pending_expired (startup) event write failed: %s", exc)
        except Exception:  # noqa: BLE001 -- boot MUST NOT block on startup prune
            # Drain failure must never block daemon startup.
            log.debug("startup prune first_turn_pending failed", exc_info=True)

        try:
            _wal = SleepWAL()
            pending = _wal.pending_entries()
            if pending:
                log.warning(
                    "daemon startup: %d pending WAL entries found — prior process may have"
                    " died mid-sleep; entries logged but NOT re-executed",
                    len(pending),
                )
                write_event(
                    store,
                    "sleep_wal_pending_recovered",
                    {"count": len(pending), "phase": "startup"},
                    severity="info",
                )
        except Exception:  # noqa: BLE001 -- WAL check MUST NOT crash boot
            log.exception("daemon startup: sleep_wal pending check failed")

        # Startup drain runs AFTER SocketServer bind (below). Rationale: a
        # malformed deferred-captures file used to re-crash the daemon every
        # boot because drain ran BEFORE the MCP socket bound, leaving
        # `iai-mcp daemon status` permanently unreachable. The drain itself
        # is scheduled as a background asyncio task — see PATCH C below.

        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                loop.add_signal_handler(sig, shutdown.set)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main-thread: no signal handlers.
                pass

        # Hippo boot health check — runs BEFORE the SocketServer binds so the
        # result event is always written before any MCP client can connect.
        # O(1): one COUNT query + len(_label_map). No compaction is needed at
        # boot for the SQLite/hnswlib backend.
        try:
            health = _hippo_health_check_on_boot(store)
            await asyncio.to_thread(
                write_event,
                store,
                "hippo_boot_health",
                health,
                severity=("info" if health.get("action") == "ok" else "warning"),
            )
        except Exception:  # noqa: BLE001
            log.debug("hippo boot health check failed", exc_info=True)

        # SocketServer is the SINGLE binder of ~/.iai-mcp/.daemon.sock.
        # concurrency.serve_control_socket has been REMOVED from this gather
        # block — both servers calling asyncio.start_unix_server on the same
        # SOCKET_PATH would EADDRINUSE on the second bind and the daemon would
        # fail to start. Backward compat for control messages is preserved
        # inside SocketServer.handle()'s dispatcher fork (jsonrpc=='2.0' →
        # core.dispatch; 'type' in CONTROL_MSG_TYPES → forward to
        # concurrency._dispatch_socket_request).
        # concurrency.serve_control_socket STAYS defined in concurrency.py
        # for test-compat.
        #
        # Full MCP-method routing over the unix control socket.
        # idle_secs defaults to env IAI_DAEMON_IDLE_SHUTDOWN_SECS or 1800.
        mcp_socket = SocketServer(store, state=state)
        mcp_socket_task = asyncio.create_task(mcp_socket.serve())
        # Yield to the event loop so the serve() task actually binds the unix
        # socket BEFORE the rest of main() proceeds with synchronous setup
        # (heartbeat scanner, idle detector, sleep pipeline, lifecycle state
        # machine). Without this yield the socket only binds on the first
        # await further down — `iai-mcp daemon status` callers would timeout
        # on cold start. 50 ms is comfortably above the bind latency on
        # macOS/Linux.
        await asyncio.sleep(0.05)

        # Off-path boot preload task. Kicks an asyncio.to_thread that
        # calls build_runtime_graph + runtime_graph_cache.save once.  When it
        # completes it sets the module-level preload_ready flag so the recall
        # loader can observe readiness.  Recall NEVER blocks on this task
        # (flag form, NOT a gating barrier — Phase-61 decoupling preserved).
        # Absent/failed preload: the loader falls back to the on-disk cache
        # (case 1/2) or labelled cold-structural-degrade (case 3).  Bounded:
        # a truly-cold store is small, so the build is fast; the expensive case
        # only arises on a warm store that already has a file → case-1 HIT.
        try:
            from iai_mcp import runtime_graph_cache as _rgc_mod

            async def _boot_preload() -> None:
                try:
                    from iai_mcp import retrieve as _retrieve_preload
                    _g, _a, _rc = await asyncio.to_thread(
                        _retrieve_preload.build_runtime_graph, store,
                    )
                    await asyncio.to_thread(
                        _rgc_mod.save, store, _a, _rc, None, 0,
                    )
                except Exception as _exc:  # noqa: BLE001 -- preload MUST NOT crash daemon
                    log.debug("boot_preload failed: %s", _exc, exc_info=True)
                finally:
                    _rgc_mod.preload_ready.set()

            asyncio.create_task(_boot_preload())
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("boot_preload scheduling failed", exc_info=True)
            try:
                import iai_mcp.runtime_graph_cache as _rgc_fallback
                _rgc_fallback.preload_ready.set()  # avoid leaving it permanently unset
            except Exception:  # noqa: BLE001
                pass

        # Background drain of deferred-captures. Runs concurrently with serve()
        # so startup never blocks on the deferred-captures queue — a malformed
        # JSONL or a long-running ingest cannot stop the MCP socket from
        # accepting connections. Per-file errors stay isolated inside
        # drain_deferred_captures; the outer try/except guarantees scheduling
        # failure does not crash the daemon.
        try:
            from iai_mcp.capture import drain_deferred_captures as _drain

            async def _drain_and_report() -> None:
                try:
                    drain_counts = await asyncio.to_thread(_drain, store)
                    if drain_counts.get("files_drained") or drain_counts.get(
                        "files_failed"
                    ):
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "deferred_drain_startup",
                            drain_counts,
                            severity="info",
                        )
                except Exception as e:  # noqa: BLE001 -- drain MUST NOT crash daemon
                    log.warning("startup deferred drain failed: %s", e, exc_info=True)
                    try:
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "deferred_drain_failed",
                            {"error": str(e)[:200], "phase": "startup"},
                            severity="warning",
                        )
                    except Exception:  # noqa: BLE001 -- event write inside boundary guard
                        log.debug("deferred_drain_failed (startup) event write failed")

            _drain_task = asyncio.create_task(_drain_and_report())
            # Attach for test introspection (no-op in production).
            try:
                mcp_socket._test_drain_task = _drain_task  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as exc:
                log.debug("test drain task attach failed: %s", exc)
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("startup drain scheduling failed", exc_info=True)

        # The `_propagate_idle_shutdown` bridge task and socket-side
        # `idle_watcher` have been removed. The lifecycle state machine takes
        # over the "idle daemon -> shut down" responsibility via the heartbeat
        # scanner + idle detector + Hibernation transition.

        # Initialise the lifecycle state machine + heartbeat scanner + idle
        # detector + sleep pipeline. The state machine reads/writes
        # ~/.iai-mcp/lifecycle_state.json via fcntl flock.
        from iai_mcp.heartbeat_scanner import HeartbeatScanner as _HeartbeatScanner
        from iai_mcp.idle_detector import IdleDetector as _IdleDetector
        from iai_mcp.lifecycle import (
            LifecycleEvent as _LifecycleEvent,
        )
        from iai_mcp.lifecycle import (
            LifecycleStateMachine as _LifecycleStateMachine,
        )
        from iai_mcp.lifecycle_state import LifecycleState as _LifecycleState
        # S2Coordinator import + the two normal-control-flow exceptions caught
        # at every dispatch call site. Imported here next to the LSM import
        # so the FSM-wire-up code sits adjacent in the dependency graph.
        from iai_mcp.s2_coordinator import (
            S2Coordinator,
            S2OscillationBlocked,
            S2OscillationConflict,
        )
        from iai_mcp.sleep_pipeline import SleepPipeline as _SleepPipeline

        # Honor IAI_MCP_STORE for the wrappers dir resolution (test isolation
        # + multi-tenant deployments). Falls back to ~/.iai-mcp/wrappers in
        # production where the env var is unset.
        from pathlib import Path as _PathHere
        _store_root = os.environ.get("IAI_MCP_STORE")
        _wrappers_dir = (
            _PathHere(_store_root) if _store_root else _PathHere.home() / ".iai-mcp"
        ) / "wrappers"
        _heartbeat_scanner = _HeartbeatScanner(_wrappers_dir)
        _idle_detector = _IdleDetector()
        _sleep_pipeline = _SleepPipeline(store=store)

        # Construct S2Coordinator first so the LifecycleStateMachine can be
        # built with it already wired. Config is CALL-ON-DEMAND; we snapshot
        # values into the constructor because daemon-runtime never mutates env
        # between boot and shutdown.
        from pathlib import Path as _PathS2
        _s2_config = _load_s2_config()
        _s2_coord = S2Coordinator(
            store=store,
            state_path=_PathS2.home() / ".iai-mcp" / "lifecycle_state.json",
            min_interval_sec=_s2_config.min_interval_sec,
            dry_run=_s2_config.dry_run,
        )

        # Construct the singleton PeriEventBuffer and register it on the
        # module-level accessor so events.write_event and capture.capture_turn
        # can reach it via get_buffer(). Maxlen comes from _load_stc_config();
        # per-call sites re-invoke _load_stc_config fresh so mid-session env
        # edits to window_sec / strong_event_types / dry_run take effect
        # without daemon restart. Only the buffer's maxlen is locked at boot
        # (deque maxlen is constructor-final).
        from iai_mcp.peri_event_buffer import PeriEventBuffer, set_buffer
        _stc_config = _load_stc_config()
        _peri_event_buffer = PeriEventBuffer(maxlen=_stc_config.peri_event_buffer_size)
        set_buffer(_peri_event_buffer)

        # State machine WITH the coordinator. dispatch is async and routes
        # current_state persistence through s2_coordinator.transition (the
        # SOLE production writer of current_state). The 4 FSM states
        # (WAKE / DROWSY / SLEEP / HIBERNATION) are unchanged.
        _state_machine = _LifecycleStateMachine(coordinator=_s2_coord)

        # If the wrapper kicked us via wake.signal AND our last persisted
        # state was HIBERNATION, dispatch WAKE_SIGNAL so the LSM
        # transitions back to WAKE atomically with the kickstart.
        if _wake_was_pending:
            try:
                await _state_machine.dispatch(
                    _LifecycleEvent.WAKE_SIGNAL,
                    reason="wake_on_signal_consumed",
                )
            except (S2OscillationConflict, S2OscillationBlocked):
                # Coordinator already emitted the matching event; both are
                # normal control flow, not errors.
                pass
            except Exception:  # noqa: BLE001 -- boot MUST NOT block on wake dispatch
                log.debug("wake signal dispatch failed", exc_info=True)

        # Dedicated bounded executor for cascade and maintenance off-loading.
        # Separate from the default asyncio to_thread pool so the embed convoy
        # (which saturates the default pool) cannot stall cascade workers.
        # max_workers=2: one active cascade + one spare for overlap at cooldown.
        global _cascade_executor
        _cascade_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="iai-cascade"
        )

        tick_task = asyncio.create_task(
            _scheduler_tick(store, state, mcp_socket=mcp_socket)
        )
        audit_task = asyncio.create_task(
            # The audit's periodic storage-optimize body runs unconditionally
            # once the cooldown gate passes; SLEEP-state coexistence is provided
            # by the lifecycle state machine.
            continuous_audit(store, shutdown)
        )
        s4_task = asyncio.create_task(
            _s4_offline_loop(store, shutdown)
        )
        # HIPPEA activation-cascade loop.
        cascade_task = asyncio.create_task(
            _hippea_cascade_loop(store, shutdown)
        )

        # CPU watchdog (observation-only).
        cpu_watchdog_task = asyncio.create_task(
            _cpu_watchdog_loop(store, shutdown)
        )

        # Liveness + memory self-watchdog. A plain daemon thread (NOT an asyncio
        # task — it must keep running when the event loop wedges, which is the
        # exact condition it detects). It probes the control socket with a full
        # status round-trip + reads system memory-pressure / own RSS, and on a
        # debounced wedge or approaching-jetsam memory trigger does a controlled
        # lock-free self-SIGKILL -> supervised respawn. Driven by a threading
        # event (a real thread cannot await the asyncio shutdown event); the
        # stop event is set from main()'s shutdown path so the daemon thread
        # exits its breakable sleep promptly on graceful shutdown.
        _watchdog_stop = threading.Event()
        watchdog_thread = threading.Thread(
            target=_liveness_watchdog,
            args=(store, _watchdog_stop),
            name="iai-liveness-watchdog",
            daemon=True,
        )
        watchdog_thread.start()

        # The rss_watchdog_task has been removed.
        # `_rss_watchdog_loop` / `_clean_shutdown_for_restart` /
        # `_should_restart` were the legacy mechanism for unbounded RSS;
        # the lifecycle state machine's Hibernation transition provides the
        # same "kill the process to drop RSS" behaviour as a natural
        # consequence of the WAKE -> DROWSY -> SLEEP -> HIBERNATION
        # progression.

        # Lifecycle TICK loop. Cadence: 30 seconds.
        # Responsibilities:
        #   1. Poll heartbeat scanner + idle detector.
        #   2. Dispatch HEARTBEAT_REFRESH / IDLE_5MIN / IDLE_30MIN events
        #      to the state machine based on observed activity.
        #   3. When state == SLEEP, run sleep_pipeline.run with an
        #      `interrupt_check` lambda that reads MCP socket activity.
        #      On natural completion, dispatch SLEEP_CYCLE_DONE so the
        #      state machine transitions to HIBERNATION.
        #   4. When state == HIBERNATION (with shadow_run=False), set
        #      the global shutdown event so main() exits gracefully.

        LIFECYCLE_TICK_INTERVAL_SEC: float = 30.0
        DROWSY_AFTER_SEC: float = float(
            os.environ.get("LIFECYCLE_DROWSY_AFTER_SEC", "300")
        )  # 5 min
        HIBERNATE_AFTER_SEC: float = float(
            os.environ.get("LIFECYCLE_HIBERNATE_AFTER_SEC", "7200")
        )  # 2 h (state machine HIBERNATION_GRACE_EXPIRED future-phase)
        SLEEP_HEARTBEAT_IDLE_SEC: float = float(
            os.environ.get("LIFECYCLE_SLEEP_HEARTBEAT_IDLE_SEC", "1800")
        )  # 30 min — for IDLE_30MIN dispatch threshold

        # Track when WAKE last had heartbeat activity; the lifecycle
        # state machine's last_activity_ts in lifecycle_state.json is
        # the persistent-side record, but we also keep a monotonic
        # baseline here for the IDLE_5MIN / IDLE_30MIN thresholds.
        _last_active_monotonic: list[float] = [time.monotonic()]
        # Previous-tick lifecycle state for WAKE -> DROWSY edge detection.
        _prev_lifecycle_state: list = [_LifecycleState.WAKE]
        # Tracks whether we have downgraded the daemon store from EXCLUSIVE
        # (boot integrity mode) to SHARED (WAKE concurrent-read mode).
        _lock_downgraded_to_shared: list[bool] = [False]

        async def lifecycle_tick() -> None:
            """Periodic lifecycle event dispatcher.

            Called every LIFECYCLE_TICK_INTERVAL_SEC seconds (30 s).
            Cancellation-safe via asyncio.wait_for(shutdown.wait(), ...).
            """
            while not shutdown.is_set():
                try:
                    await asyncio.wait_for(
                        shutdown.wait(),
                        timeout=LIFECYCLE_TICK_INTERVAL_SEC,
                    )
                    return  # shutdown fired
                except asyncio.TimeoutError:
                    pass

                try:
                    # 1. Probe heartbeat scanner + idle detector.
                    scanner_active = await asyncio.to_thread(
                        _heartbeat_scanner.is_active,
                    )
                    heartbeat_idle = await asyncio.to_thread(
                        _heartbeat_scanner.heartbeat_idle_30min,
                    )
                    sleep_eligible = await asyncio.to_thread(
                        _idle_detector.sleep_eligible, heartbeat_idle,
                    )

                    now_mono = time.monotonic()
                    idle_elapsed = now_mono - _last_active_monotonic[0]

                    # 0. FORCE_SLEEP dispatch — BEFORE idle/heartbeat branch so a
                    #    pending force-rem / user-sleep is not pre-empted by a
                    #    HEARTBEAT_REFRESH tick. Routes * -> DROWSY first (so the
                    #    DROWSY-edge teardown / drain runs), then DROWSY -> SLEEP.
                    #    The flag is consumed by lifecycle.py's FORCE_SLEEP transitions
                    #    and cleared here once SLEEP is reached.
                    try:
                        from iai_mcp.daemon_state import load_state as _load_ds
                        _ds = await asyncio.to_thread(_load_ds)
                        _force_rem = bool((_ds.get("force_rem_request") or {}).get("pending"))
                        _user_sleep = bool((_ds.get("user_sleep_request") or {}).get("pending"))
                        if _force_rem or _user_sleep:
                            # Dispatch FORCE_SLEEP: * -> DROWSY -> SLEEP in two hops.
                            # First hop: any state -> DROWSY.
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.FORCE_SLEEP,
                                    reason="force_sleep_request",
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass
                            # Second hop (only if now DROWSY): DROWSY -> SLEEP.
                            if _state_machine.current_state is _LifecycleState.DROWSY:
                                try:
                                    await _state_machine.dispatch(
                                        _LifecycleEvent.FORCE_SLEEP,
                                        reason="force_sleep_drowsy_to_sleep",
                                    )
                                except (S2OscillationConflict, S2OscillationBlocked):
                                    pass
                            # Clear the pending flags once SLEEP is reached.
                            if _state_machine.current_state is _LifecycleState.SLEEP:
                                _now_iso = __import__("datetime").datetime.now(
                                    __import__("datetime").timezone.utc,
                                ).isoformat()
                                _ds_upd = dict(_ds)
                                if _force_rem:
                                    req = dict(_ds_upd.get("force_rem_request") or {})
                                    req["pending"] = False
                                    req["honored_at"] = _now_iso
                                    _ds_upd["force_rem_request"] = req
                                if _user_sleep:
                                    req = dict(_ds_upd.get("user_sleep_request") or {})
                                    req["pending"] = False
                                    req["honored_at"] = _now_iso
                                    _ds_upd["user_sleep_request"] = req
                                from iai_mcp.daemon_state import save_state as _save_ds
                                await asyncio.to_thread(_save_ds, _ds_upd)
                    except Exception:  # noqa: BLE001 -- FORCE_SLEEP dispatch is best-effort
                        log.debug("lifecycle_tick FORCE_SLEEP dispatch failed", exc_info=True)

                    # Per-tick runtime fsm reconcile: keep legacy .daemon-state.json
                    # fsm_state in sync with canonical lifecycle (one-directional
                    # canonical -> legacy) so doctor / topology readers stay correct.
                    try:
                        from iai_mcp.fsm_reconcile import reconcile_fsm_state
                        reconcile_fsm_state(auto_correct=True)
                    except Exception:  # noqa: BLE001 -- reconcile is best-effort
                        pass

                    if scanner_active:
                        # Wrapper is alive — refresh activity baseline
                        # and dispatch HEARTBEAT_REFRESH (DROWSY -> WAKE).
                        _last_active_monotonic[0] = now_mono
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.HEARTBEAT_REFRESH,
                                reason="heartbeat_refresh_active_wrapper",
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass
                    elif idle_elapsed >= SLEEP_HEARTBEAT_IDLE_SEC and sleep_eligible:
                        # 30 min idle + hardware confirmation → request
                        # SLEEP transition. Payload guard satisfies the
                        # transition-table requirement.
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.IDLE_30MIN,
                                reason="sleep_on_idle_30min",
                                sleep_eligible=True,
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass
                    elif idle_elapsed >= DROWSY_AFTER_SEC:
                        # 5 min idle → DROWSY (no-op if already there).
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.IDLE_5MIN,
                                reason="drowsy_on_idle_5min",
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass

                    # 2. If state is now SLEEP, run the sleep pipeline
                    #    with bounded deferral.
                    current = _state_machine.current_state
                    # WAKE -> DROWSY edge: drain deferred captures once per
                    # entry. Guarded by _prev_lifecycle_state so consecutive
                    # DROWSY ticks do not re-fire.
                    if _should_drain_on_drowsy_edge(_prev_lifecycle_state[0], current):
                        try:
                            from iai_mcp.capture import drain_deferred_captures

                            await asyncio.to_thread(
                                _run_drowsy_drain,
                                store,
                                drain_fn=drain_deferred_captures,
                                write_event_fn=write_event,
                            )
                        except Exception:  # noqa: BLE001 -- drowsy drain non-fatal
                            log.debug("lifecycle_tick drowsy drain failed", exc_info=True)

                        # Ordered wake sequence: re-embed pending rows (1), ingest
                        # sidecars (2), rebuild index + graph-cache refresh (3).
                        # Gated inside pending_embeddings_wake_sequence behind a
                        # dirty-check so an idle wake is near-free.
                        try:
                            from iai_mcp.embed import embedder_for_store
                            from iai_mcp import runtime_graph_cache as _rgc

                            def _run_wake_sequence():
                                try:
                                    _emb = embedder_for_store(store)
                                except Exception:
                                    _emb = None
                                result = store.db.pending_embeddings_wake_sequence(embedder=_emb)
                                if result.get("action") != "skip":
                                    # Step 3: graph-cache invalidation so a
                                    # re-embedded row becomes a warm semantic hit.
                                    try:
                                        _rgc.invalidate(store)
                                    except Exception:
                                        pass
                                return result

                            _wake_seq_result = await asyncio.to_thread(_run_wake_sequence)
                            # Rebuild the graph cache in the background so a
                            # re-embedded row becomes a warm semantic hit on the
                            # next recall; recall never blocks on this rebuild
                            # (flag-not-gate design — daemon never a gatekeeper).
                            if (
                                isinstance(_wake_seq_result, dict)
                                and _wake_seq_result.get("action") != "skip"
                            ):
                                try:
                                    _kick_drowsy_rgc_rebuild(store)
                                except Exception:  # noqa: BLE001 -- best-effort
                                    log.debug("drowsy-edge kick failed", exc_info=True)
                        except Exception:  # noqa: BLE001 -- wake sequence non-fatal
                            log.debug("lifecycle_tick pending_embeddings_wake_sequence failed", exc_info=True)
                    # Downgrade EX → SH on first WAKE entry so concurrent
                    # SHARED clients can open the store while the daemon is awake.
                    if (
                        not _lock_downgraded_to_shared[0]
                        and current in (
                            _LifecycleState.WAKE,
                            _LifecycleState.DROWSY,
                        )
                    ):
                        try:
                            await asyncio.to_thread(store.db.downgrade_to_shared)
                            _lock_downgraded_to_shared[0] = True
                            log.debug("daemon_lock_downgrade: EX→SH on first WAKE entry")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_downgrade failed", exc_info=True)

                    _prev_lifecycle_state[0] = current
                    if current is _LifecycleState.SLEEP:
                        def _interrupt_check() -> bool:
                            # Bounded deferral: fire the interrupt if
                            # MCP traffic is active or recent.
                            if mcp_socket.active_connections > 0:
                                return True
                            elapsed = (
                                time.monotonic() - mcp_socket.last_activity_ts
                            )
                            return elapsed < INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC

                        # Escalate SH → EX for the consolidation window.
                        # Clients see the intent flag and back off.
                        try:
                            await asyncio.to_thread(store.db.escalate_to_exclusive)
                            log.debug("daemon_lock_escalate: SH→EX for sleep pipeline")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_escalate failed", exc_info=True)

                        result = await asyncio.to_thread(
                            _sleep_pipeline.run, _interrupt_check,
                        )

                        # --- WAKE hook (UNDER LOCK_EX, BEFORE downgrade) ---
                        # Run the 4 legacy-unique WAKE-side outputs while still
                        # holding the exclusive lock. This ensures _write_session_start_cache
                        # reflects the just-consolidated store and no client can read a stale
                        # precache in the window between downgrade and cache write.
                        # Runs even on interrupted/failed pipeline (best-effort each).
                        # Session-start precache (~3000-token invariant).
                        try:
                            await asyncio.to_thread(_write_session_start_cache, store)
                        except Exception:  # noqa: BLE001 -- precache MUST NOT crash
                            log.debug("lifecycle_tick _write_session_start_cache failed", exc_info=True)
                        # Processed top-N salience write.
                        try:
                            from iai_mcp.memory_bank import write_processed_salience_top_n
                            await asyncio.to_thread(write_processed_salience_top_n, store)
                        except (ImportError, OSError, ValueError, RuntimeError) as exc:
                            log.debug("lifecycle_tick write_processed_salience_top_n failed: %s", exc)
                        # Drain still-open other-session live capture files.
                        try:
                            from iai_mcp.capture import drain_active_live_captures
                            _live_drain = await asyncio.to_thread(
                                drain_active_live_captures, store, exclude_session_id="-",
                            )
                            if _live_drain.get("events_inserted"):
                                await asyncio.to_thread(
                                    write_event, store, "active_live_drain_wake",
                                    _live_drain, severity="info",
                                )
                        except Exception as _exc:  # noqa: BLE001 -- drain MUST NOT crash
                            log.debug("lifecycle_tick active_live_drain failed: %s", _exc)
                        # Flush deferred-provenance buffer.
                        try:
                            from iai_mcp.provenance_buffer import flush_deferred_provenance
                            _prov_count = await asyncio.to_thread(
                                flush_deferred_provenance, store,
                            )
                            if _prov_count > 0:
                                await asyncio.to_thread(
                                    write_event, store, "deferred_provenance_flush_wake",
                                    {"count": _prov_count}, severity="info",
                                )
                        except Exception as _exc:  # noqa: BLE001 -- flush MUST NOT crash
                            log.debug("lifecycle_tick flush_deferred_provenance failed: %s", _exc)
                        # Rebuild graph cache if cold after an interrupted
                        # consolidation cycle (the topology rebuild step may have
                        # been skipped).  Runs under EX before downgrade so the
                        # write is exclusive.  Skips when the cache is already
                        # overlay/normal — avoids unnecessary rebuild and extra
                        # exclusive-lock time.  Best-effort: never crashes the hook.
                        try:
                            await asyncio.to_thread(_wake_hook_rebuild_if_cold, store)
                        except Exception as _exc:  # noqa: BLE001 -- best-effort
                            log.debug("lifecycle_tick wake-hook rebuild-if-cold failed: %s", _exc)

                        # Downgrade EX → SH after the consolidation window.
                        try:
                            await asyncio.to_thread(store.db.downgrade_to_shared)
                            log.debug("daemon_lock_downgrade: EX→SH after sleep pipeline")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_downgrade_post_sleep failed", exc_info=True)
                        if (
                            not result.get("interrupted", False)
                            and result.get("failed_step") is None
                            and not result.get("quarantine_triggered", False)
                            and len(result.get("completed_steps", [])) >= 5
                        ):
                            # Natural completion of all 5 steps → maybe
                            # transition to HIBERNATION.
                            # `still_idle` payload guard: re-check idle
                            # AFTER the pipeline ran (it may have run
                            # for several seconds; user activity may
                            # have arrived in between).
                            still_idle_now = await asyncio.to_thread(
                                _heartbeat_scanner.heartbeat_idle_30min,
                            )
                            sleep_eligible_now = await asyncio.to_thread(
                                _idle_detector.sleep_eligible, still_idle_now,
                            )
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.SLEEP_CYCLE_DONE,
                                    reason="hibernate_on_sleep_cycle_done",
                                    still_idle=(still_idle_now and sleep_eligible_now),
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass

                    # 3. If state is HIBERNATION and shadow_run=False,
                    #    set the global shutdown event. main()'s finally
                    #    block will release the lifecycle lock and exit 0.
                    current = _state_machine.current_state
                    if (
                        current is _LifecycleState.HIBERNATION
                        and not _state_machine.shadow_run
                    ):
                        try:
                            write_event(
                                store,
                                "lifecycle_hibernation_exit",
                                {
                                    "reason": "lifecycle_tick_hibernation",
                                    "shadow_run": False,
                                },
                                severity="info",
                            )
                        except (OSError, RuntimeError) as exc:
                            log.debug("lifecycle_hibernation_exit event write failed: %s", exc)
                        shutdown.set()
                        return
                except Exception:  # noqa: BLE001 -- lifecycle tick must NEVER crash
                    # Defensive: any error in the lifecycle tick should
                    # not bring down the daemon. The next tick gets a
                    # fresh chance.
                    log.warning("lifecycle tick iteration failed", exc_info=True)

        lifecycle_tick_task = asyncio.create_task(lifecycle_tick())

        try:
            await shutdown.wait()
        finally:
            # Simplified shutdown set.
            # `idle_propagator_task` and `rss_watchdog_task` are gone; the
            # remaining 6 tasks (mcp_socket + tick + audit + s4 + cascade
            # + cpu_watchdog) form the cancel set. Trigger SocketServer's
            # graceful drain explicitly so connections close before the
            # asyncio.Server is torn down by task cancellation.
            try:
                mcp_socket.shutdown_event.set()
            except (AttributeError, RuntimeError) as exc:
                log.debug("mcp_socket shutdown_event.set failed: %s", exc)
            # Signal the liveness watchdog thread to break out of its sleep and
            # exit. It is a daemon thread (won't block process exit) but we set
            # the stop event so graceful shutdown is clean.
            try:
                _watchdog_stop.set()
            except (NameError, RuntimeError) as exc:
                log.debug("watchdog stop set failed: %s", exc)
            # Shut down the dedicated cascade executor. wait=False so the
            # shutdown path is not blocked by in-flight cascade work.
            try:
                if _cascade_executor is not None:
                    _cascade_executor.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                log.debug("cascade executor shutdown failed: %s", exc)
            _cancel_targets = [
                tick_task, audit_task, s4_task, cascade_task,
                mcp_socket_task,
                cpu_watchdog_task,
                lifecycle_tick_task,
            ]
            for t in _cancel_targets:
                t.cancel()
            # Drain task exceptions silently: we're shutting down.
            await asyncio.gather(*_cancel_targets, return_exceptions=True)
            # Graceful-shutdown flush. Catches the
            # sub-threshold tail of buffered events on SIGINT / SIGTERM before
            # the lifecycle lock releases. Synchronous call (no asyncio.to_thread)
            # because by this point the asyncio event loop is winding down.
            # flush_event_buffer is sync-safe (synchronous implementation).
            try:
                from iai_mcp.events import flush_event_buffer

                events_count = flush_event_buffer(store)
                if events_count > 0:
                    log.info("events buffer flushed on shutdown: count=%d", events_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("events buffer shutdown flush failed: %s", e, exc_info=True)
            # Graceful-shutdown records flush — catches the sub-threshold tail of
            # buffered records on SIGINT / SIGTERM before the lifecycle lock releases.
            # Synchronous (no asyncio.to_thread) because the asyncio loop is winding down.
            try:
                from iai_mcp.store import flush_record_buffer

                records_count = flush_record_buffer(store)
                if records_count > 0:
                    log.info("records buffer flushed on shutdown: count=%d", records_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("records buffer shutdown flush failed: %s", e, exc_info=True)
            # Graceful-shutdown edges flush — synchronous (asyncio loop is winding down).
            try:
                from iai_mcp.store import flush_edge_buffer

                edges_count = flush_edge_buffer(store)
                if edges_count > 0:
                    log.info("edges buffer flushed on shutdown: count=%d", edges_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("edges buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                write_event(store, "daemon_stopped", {"state": state.get("fsm_state")})
            except (OSError, RuntimeError) as exc:
                log.debug("daemon_stopped event write failed: %s", exc)
            # Persist final state so next boot sees a clean shutdown marker.
            # Clear the on-disk
            # user_requested_shutdown sentinel so it does not leak across
            # boots. Exit code is uniformly 0 — the plist's KeepAlive=
            # {"Crashed": true} ensures graceful 0 stays dead until wrapper
            # kickstart.
            _clear_user_shutdown_sentinel(state)
            try:
                state.pop("daemon_pid", None)
                state["daemon_stopped_at"] = datetime.now(timezone.utc).isoformat()
                await asyncio.to_thread(save_state, state)
            except (OSError, ValueError) as exc:
                log.debug("final save_state failed: %s", exc)
            # Release the lifecycle lockfile so the next daemon boot can acquire
            # cleanly. release() is idempotent.
            try:
                lifecycle_lock.release()
            except (OSError, RuntimeError) as exc:
                log.debug("lifecycle_lock release failed: %s", exc)
    finally:
        # Restore the original embedder funnel so the held singleton does not
        # outlive this process's serving lifetime. Guarded by the sentinel: a
        # build/hold failure (override never set) or an early lock-conflict
        # return (install never reached) restores nothing. Runs on clean
        # shutdown, on the hibernation-transition return, and on any raised
        # exception from the post-install boot region. Internally crash-safe.
        _restore_embedder_funnel(_orig_efs, _override_installed)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
