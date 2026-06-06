"""Persist Leiden community assignment + rich-club to disk so the first
``memory_recall`` call in a fresh core process does not rebuild these
expensive artefacts from scratch.

The naive path rebuilds everything on every call:

    graph   = MemoryGraph()          # ~100 ms to construct from rows
    detect_communities(graph)        # Leiden, ~200 ms at N=1k
    rich_club_nodes(graph, 0.10)     # ~20 ms

The cold path runs ~440 ms at N=1k. Caching the *Leiden output* and
the rich-club node list eliminates the two expensive computations when
the store has not changed. MemoryGraph construction itself is cheap
enough to rebuild per call; caching it too would require pickle (the
graph is not JSON-friendly) and the security-vs-speed
trade-off is not worth it for ~100 ms.

**Invalidation** — any of these triggers a rebuild:

- Record count changed (user saved / consolidated / merged)
- Edge count changed (Hebbian reinforcement or contradiction added)
- SCHEMA_VERSION_CURRENT bumped (store migrated)
- store.embed_dim changed (user swapped embedder)
- CACHE_VERSION bumped (this module's on-disk format changed)

Any inconsistency — corrupt JSON, unreadable file, unknown keys —
falls through to a clean rebuild. The cache is purely an optimisation;
the authoritative graph is always the SQLite store.

**Write strategy**: every ``save()`` writes a ``.tmp`` file first then
``os.replace``s it over the real path — atomic on POSIX. A crash
mid-write leaves either the old cache intact or no cache at all;
never a partially written file. No flush timer; the cache refreshes
on the next ``build_runtime_graph`` call when the key changes.

**Why JSON not pickle**: the cached payload is list-of-UUIDs,
list-of-floats and scalars — all JSON-native after simple UUID→str
conversion. JSON avoids the arbitrary-code-execution risk of pickle
and makes the cache auditable (a user can cat the file to see what
the brain thinks its communities are).

Invariants:

- Zero network calls: pure local JSON + filesystem operations.
- Read-only against the store: cache writes go to the cache file
  only, never to any store table.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from cryptography.exceptions import InvalidTag

logger = logging.getLogger(__name__)

from iai_mcp.crypto import (
    CryptoKey,
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import SCHEMA_VERSION_CURRENT

# Daemon boot preload readiness flag (threading.Event, process-wide).
# The daemon sets this after the boot preload task completes (build_runtime_graph
# + cache save).  core.py's loader reads it for observability/labelling only —
# correctness comes from the on-disk cache file, not this flag.
# Absent/not-set means daemon-down or preload not yet done; the 3-case loader
# still serves correctly from the cache file (case 1/2) or labels cold-degrade
# (case 3).  Never blocks recall (flag form, NOT a gating barrier).
preload_ready: threading.Event = threading.Event()

# Background-rebuild completion flag (threading.Event, process-wide).
# Mirrors ``preload_ready`` above.  The daemon sets this (in a ``finally``)
# after the DROWSY-edge background rebuild completes so callers can
# synchronise on real completion.  The helper ``_rebuild_and_save_rgc``
# itself does NOT set this flag — the daemon caller owns the Event so the
# same helper can be reused by the nightly sleep step without touching it.
# Never blocks recall (flag form, NOT a gating barrier).
rebuild_ready: threading.Event = threading.Event()


# Bump this whenever the on-disk cache shape changes. A mismatch
# forces every user on the old shape to rebuild -- safer than silently
# loading a file whose key contract has drifted.
#
# Version history:
#   "06-02-v1": payload carries max_degree (one int) so the rank stage
#     can normalise log(1+deg) by log(1+max_deg) without re-walking the
#     live graph on every recall.
#   "07-09-v3": cache file is now AES-256-GCM-wrapped. Old "06-02-v1"
#     caches are treated as legacy plaintext: read once, lazily re-saved
#     as ciphertext on first warm-start.
#   "62-04-v4": staleness window applied to records_count and edges_count
#     key components so single ambient writes do not invalidate the cache
#     on the recall hot path.
#   "62-02-v5": derived RecallIndex overlay — generation epoch
#     (nightly-stamp only, no per-mutation bump) + freshness-fuse baseline
#     (rebuild_timestamp + dirty_counter reset) added to snapshot payload.
#     Old "62-04-v4" caches lack these fields and are rejected to prevent
#     a stale-format snapshot being served under the new epoch semantics.
CACHE_VERSION: str = "62-02-v5"

# Staleness window for records_count and edges_count in _cache_key.
# Only the WINDOW-divided floor enters the key, so writes within a window
# (< STALENESS_WINDOW new records or edges) keep the same key → a single
# ambient write yields a try_load HIT instead of a topology-MISS rebuild.
# SCHEMA_VERSION, embed_dim, and CACHE_VERSION are still exact components.
# Sized so a meaningful topology shift (e.g. 10 new records) still forces
# a rebuild at the next window boundary.
_STALENESS_WINDOW: int = 10
LEGACY_CACHE_VERSION_PLAINTEXT: str = "06-02-v1"

# AES-GCM associated data (AD): binds the ciphertext to this format and
# version. A bytewise tampering attempt that swaps the file with a
# v06-02-v1 plaintext or any other stream fails the decrypt tag check.
_CACHE_AAD: bytes = b"runtime-graph-cache:v3"

CACHE_FILENAME: str = "runtime_graph_cache.json"

# ---------------------------------------------------------------------------
# Derived RecallIndex overlay (O(1) freshness fuse)
# ---------------------------------------------------------------------------

# Maximum wall-time age (seconds) since the last nightly rebuild before the
# overlay freshness fuse trips to the Layer-1 bypass.  25 hours covers a
# full day's drift and one missed nightly window; the nightly rebuild resets
# the wall-clock baseline to zero.  Chosen so a structurally-stale GLOBAL
# bias (mosaic communities + rich-club cores stable across single-day deltas)
# is never silently served past this window.
_FUSE_MAX_AGE_SECONDS: float = 25.0 * 3600.0

# Maximum in-process record-mutation count before the overlay freshness fuse
# trips.  Counts only RECORD inserts/updates/deletes visible to the
# register_graph_sync_hook (RECORD-only — recall-path boost_edges are
# invisible and do NOT trip the fuse).  Reset to zero by
# the nightly rebuild step.  Sized at 50: a day of typical ambient captures
# (~10-30) stays well under this; a bulk import or heavy multi-session day
# trips the fuse and forces a bypass to the last-good snapshot.
_FUSE_DIRTY_THRESHOLD: int = 50

# In-process record-mutation counter for the O(1) freshness fuse.
# Incremented by the COMPOSED register_graph_sync_hook (write-path only —
# not the recall hot path).  Reset to zero by the nightly rebuild step.
# Module-global (process-wide singleton); thread-safe via _DIRTY_COUNTER_LOCK.
_dirty_counter: int = 0
_DIRTY_COUNTER_LOCK = threading.Lock()


def increment_dirty_counter() -> None:
    """Increment the in-process record-mutation dirty counter (O(1)).

    Called exclusively from the COMPOSED register_graph_sync_hook in
    retrieve.py on every record insert/update/delete.  NEVER called from
    the recall hot path; recall-path boost_edges are invisible to this
    counter.
    """
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter += 1


def reset_dirty_counter() -> None:
    """Reset the in-process record-mutation dirty counter to zero.

    Called by the nightly RecallIndex rebuild step after it stamps a fresh
    generation epoch and rebuild_timestamp onto the snapshot.  The next
    fuse evaluation will measure age/delta from the fresh baseline.
    """
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter = 0


def get_dirty_counter() -> int:
    """Return the current dirty counter value (O(1), no store access)."""
    with _DIRTY_COUNTER_LOCK:
        return _dirty_counter


# Size cap for the on-disk cache. When the encoded payload exceeds this,
# ``save`` drops ``node_payload`` (the large per-record embedding map) and
# writes only ``assignment + rich_club``. Cold-start ``build_runtime_graph``
# rehydrates the node payload from the store on the next recall;
# the cache remains advisory. 10 MiB holds the Leiden + rich-club artefacts
# for a ~50k-record store comfortably while keeping cold-start load under
# the session-start token budget.
MAX_CACHE_BYTES: int = 10 * 1024 * 1024


def _cache_path(store: Any) -> Path:
    """Cache file lives next to the store directory so it travels with
    the store on backup / move. One cache file per MemoryStore."""
    root = getattr(store, "root", None)
    if root is None:
        root = Path.cwd()
    return Path(root) / CACHE_FILENAME


def _cache_encryption_key(store: Any) -> bytes:
    """32-byte AES key for the runtime-graph-cache sidecar. Reuses the
    store's already-cached key whenever possible to
    avoid a second keyring round-trip. Falls back to a fresh CryptoKey
    lookup keyed on the store's user_id (or "default") when the store
    doesn't expose a cached key — the same passphrase / keyring contract
    applies, so the resolved key is identical.
    """
    # MemoryStore caches its key after the first encryption call
    # (store.py:_key()); that's the cheapest path. Defensive getattr
    # so this module stays usable from non-store call sites in tests.
    cached_via_store = getattr(store, "_crypto_key", None)
    if isinstance(cached_via_store, (bytes, bytearray)) and len(cached_via_store) == 32:
        return bytes(cached_via_store)
    if hasattr(store, "_key") and callable(store._key):
        try:
            key = store._key()
            if isinstance(key, (bytes, bytearray)) and len(key) == 32:
                return bytes(key)
        except (OSError, ValueError, RuntimeError):
            pass
    user_id = getattr(store, "user_id", "default") or "default"
    return CryptoKey(user_id=user_id).get_or_create()


def _cache_key(store: Any) -> tuple:
    """Monotonic identity for "the cached graph is still correct for this
    store state". Returns a tuple whose components are used for exact
    comparison by try_load and for parity checks by load_last_good_structural.

    Key shape: (records_window, edges_window, schema_version, embed_dim, cache_version)

    records_window and edges_window are staleness-windowed (floor-divided by
    _STALENESS_WINDOW) so single ambient writes stay within the same window →
    a single record or edge write does not invalidate the cache key on the
    recall hot path.  SCHEMA_VERSION_CURRENT, embed_dim, and CACHE_VERSION are
    exact components; any change to them invalidates the cache immediately.
    """
    # Use the non-pending count so the cache key stays stable while
    # pending rows exist.  The retrieve.py MISS-path walk skips pending rows;
    # both the gate and this key must agree on the same count.
    try:
        records_count = int(store.active_records_count())
    except (OSError, ValueError, KeyError, AttributeError):
        try:
            records_count = int(store.db.open_table("records").count_rows())
        except (OSError, ValueError, KeyError, AttributeError):
            records_count = -1
    try:
        edges_count = int(store.db.open_table("edges").count_rows())
    except (OSError, ValueError, KeyError, AttributeError):
        edges_count = -1
    embed_dim = int(getattr(store, "embed_dim", 0))
    # Floor-divide count components by the staleness window so single writes
    # stay within the same window bucket and do not force a cache MISS.
    rc_window = records_count // _STALENESS_WINDOW if records_count >= 0 else records_count
    ec_window = edges_count // _STALENESS_WINDOW if edges_count >= 0 else edges_count
    return (
        rc_window,
        ec_window,
        SCHEMA_VERSION_CURRENT,
        embed_dim,
        CACHE_VERSION,
    )


def _parity_components(store: Any) -> tuple:
    """Count-FREE parity triple used by the overlay consult and
    load_last_good_structural to enforce schema/embed_dim/cache_version
    invariants WITHOUT any store.active_records_count() / count_rows()
    call on the recall hot path.

    Returns (schema_version, embed_dim, cache_version) — matches
    key-tuple positions 2, 3, 4 of _cache_key().
    """
    embed_dim = int(getattr(store, "embed_dim", 0))
    return (SCHEMA_VERSION_CURRENT, embed_dim, CACHE_VERSION)


# Sentinel returned by consult_overlay to signal a typed bypass.
# The caller routes to load_last_good_structural; it is NEVER a hot-path
# rebuild / detect_communities call.
class _OverlayBypass:
    """Typed sentinel for overlay fuse-trip / epoch-mismatch / invariant failure.

    ``reason`` is one of "epoch_mismatch" | "fuse_tripped" | "invariant_failure"
    | "no_snapshot" | "parity_mismatch".  Only "fuse_tripped" emits the
    freshness_fuse_tripped telemetry event.

    ``age_ms`` is populated for fuse_tripped only (wall-time since rebuild).
    """
    __slots__ = ("reason", "age_ms")

    def __init__(self, reason: str, age_ms: int = 0) -> None:
        self.reason = reason
        self.age_ms = age_ms

    def __repr__(self) -> str:  # pragma: no cover
        return f"_OverlayBypass(reason={self.reason!r}, age_ms={self.age_ms})"


def _check_snapshot_invariants(data: dict) -> bool:
    """Snapshot-internal invariant checks.

    All checks are derived from the decoded snapshot itself — zero store
    calls.  Returns True iff the snapshot passes all invariants.

    Invariants:
    1. Community count is in a sane band (1 .. 100000 — not empty, not
       pathologically exploded).
    2. rich_club ids are a subset of node ids (no dangling hub reference).
    3. modularity is in [-1.0, 1.0].
    """
    assignment_raw = data.get("assignment")
    if not isinstance(assignment_raw, dict):
        return False
    node_to_community = assignment_raw.get("node_to_community") or {}
    if not isinstance(node_to_community, dict):
        return False
    # Invariant 1: sane community count
    n_communities = len(set(node_to_community.values()))
    if n_communities == 0 and len(node_to_community) > 0:
        return False
    if n_communities > 100_000:
        return False
    # Invariant 2: rich_club ⊆ node ids
    rich_club_raw = data.get("rich_club") or []
    if isinstance(rich_club_raw, list) and rich_club_raw:
        node_ids = set(node_to_community.keys())
        for rc_id in rich_club_raw:
            if rc_id not in node_ids:
                return False
    # Invariant 3: sane modularity
    try:
        modularity = float(assignment_raw.get("modularity", 0.0) or 0.0)
        if not (-1.0 <= modularity <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def consult_overlay(store: Any) -> "tuple | _OverlayBypass":
    """O(1) derived RecallIndex overlay consult.

    Reads the on-disk snapshot (AES-GCM decrypt) and evaluates:
    1. Parity: schema_version + embed_dim + cache_version match (count-FREE).
    2. Generation epoch: snapshot's stamped epoch matches the current epoch.
    3. Freshness fuse: wall-time age since rebuild_timestamp <=
       _FUSE_MAX_AGE_SECONDS AND in-process dirty counter <= _FUSE_DIRTY_THRESHOLD.
       Both checks are O(1) — no SELECT count(*) / active_records_count /
       count_rows call.
    4. Snapshot-internal invariants: community count sane, rich_club ⊆ node ids.

    On any mismatch or fuse-trip returns an _OverlayBypass sentinel;
    the caller must route to load_last_good_structural (the Layer-1 last-good
    bypass) — NEVER to build_runtime_graph / detect_communities on the hot path.

    On success returns (assignment, rich_club) from the snapshot (O(1) HIT).

    Emits freshness_fuse_tripped telemetry on fuse-trip only (reason
    "fuse_tripped"), with age_ms and pending_rebuild=False, so the
    accept-stable-global-bias-intra-day choice is OBSERVABLE.
    """
    data = _load_and_decrypt_cache(store)
    if data is None:
        return _OverlayBypass("no_snapshot")

    # 0. Cache-version check (parity component 3).
    if data.get("cache_version") != CACHE_VERSION:
        return _OverlayBypass("parity_mismatch")

    # 1. Count-free parity triple.
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 5:
        return _OverlayBypass("parity_mismatch")
    current_parity = _parity_components(store)
    if saved_key[2] != current_parity[0]:  # schema_version
        return _OverlayBypass("parity_mismatch")
    if saved_key[3] != current_parity[1]:  # embed_dim
        return _OverlayBypass("parity_mismatch")
    if saved_key[4] != current_parity[2]:  # cache_version
        return _OverlayBypass("parity_mismatch")

    # 2. Generation epoch: snapshot's generation must match stored epoch.
    # The snapshot carries "generation" (int monotonic counter stamped by
    # the nightly rebuild).  The current epoch is read O(1) from the module-
    # level counter — zero store calls.
    snapshot_generation = data.get("generation", 0)
    if not isinstance(snapshot_generation, int):
        return _OverlayBypass("epoch_mismatch")
    # The authoritative current generation IS the snapshot's own stamped value
    # when the snapshot is the most recently written one.  The epoch comparison
    # is against the process-level _current_generation counter which is updated
    # by save_with_generation() on every nightly rebuild (and may drift
    # ahead if the snapshot on disk belongs to an older cycle).
    # We compare: if the snapshot was written by a nightly rebuild with a
    # higher-epoch stamp than the module knows about, we load it and update.
    # The key invariant: a snapshot written by THIS process's last nightly
    # rebuild has epoch == _current_generation.  A snapshot from a PRIOR
    # nightly cycle that was never updated has epoch < _current_generation.
    current_gen = get_current_generation()
    # generation == 0 means no nightly rebuild has stamped this snapshot — the
    # overlay should not serve it (bypass to Layer-1 so existing 62-04 / 62-05
    # code paths are preserved until the first nightly rebuild runs).
    if current_gen == 0 or snapshot_generation == 0 or snapshot_generation != current_gen:
        return _OverlayBypass("epoch_mismatch")

    # 3. Freshness fuse — O(1), no store access.
    rebuild_ts_str = data.get("rebuild_timestamp")
    age_ms = 0
    if rebuild_ts_str:
        try:
            rebuild_dt = datetime.fromisoformat(str(rebuild_ts_str))
            if rebuild_dt.tzinfo is None:
                rebuild_dt = rebuild_dt.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - rebuild_dt).total_seconds()
            age_ms = max(0, int(age_sec * 1000))
        except (TypeError, ValueError):
            age_sec = _FUSE_MAX_AGE_SECONDS + 1.0  # treat unparseable as expired
            age_ms = int(age_sec * 1000)
    else:
        # No rebuild_timestamp: snapshot was written by a non-nightly path
        # (e.g. build_runtime_graph on cache miss — the freshest possible ground
        # truth since it was just derived from the live store).  Treat as age=0
        # (no max_age trip) so the dirty counter alone gates freshness for these.
        age_sec = 0.0
        age_ms = 0

    dirty = get_dirty_counter()
    if age_sec > _FUSE_MAX_AGE_SECONDS or dirty > _FUSE_DIRTY_THRESHOLD:
        # Fuse tripped.  Emit telemetry so the accept-stale choice is observable.
        _emit_freshness_fuse_tripped(store, age_ms=age_ms)
        return _OverlayBypass("fuse_tripped", age_ms=age_ms)

    # 4. Snapshot-internal invariants.
    if not _check_snapshot_invariants(data):
        return _OverlayBypass("invariant_failure")

    # All checks passed: decode and serve from overlay (O(1) HIT).
    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache overlay decode failed: %s", exc)
        return _OverlayBypass("invariant_failure")

    return assignment, rich_club


def _emit_freshness_fuse_tripped(store: Any, *, age_ms: int) -> None:
    """Emit the freshness_fuse_tripped telemetry event (Option-a observability).

    Buffered so a sustained post-threshold period does not spam the events
    table on every recall.  Never raises — observability must not break recall.
    """
    try:
        from iai_mcp.events import (
            TELEMETRY_FRESHNESS_FUSE_TRIPPED,
            write_event,
        )
        from iai_mcp.store import MemoryStore

        if not isinstance(store, MemoryStore):
            return
        write_event(
            store,
            TELEMETRY_FRESHNESS_FUSE_TRIPPED,
            {"age_ms": int(age_ms), "pending_rebuild": False},
            severity="info",
            buffered=True,
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break recall
        pass


# ---------------------------------------------------------------------------
# Generation epoch (O(1) in-process counter, nightly-stamp only)
# ---------------------------------------------------------------------------

# In-process monotonic generation counter.  Stamped ONLY by the nightly
# RecallIndex rebuild step (sleep_pipeline.py).  Never bumped per mutation —
# that would self-invalidate the overlay on recall-path boost_edges writes.
# When the daemon restarts or the module is first imported,
# the counter is 0 until the nightly rebuild stamps a fresh value OR the
# snapshot on disk carries a stamped epoch that is read back via
# load_current_generation_from_snapshot().
_current_generation: int = 0
_GEN_LOCK = threading.Lock()


def get_current_generation() -> int:
    """Return the current in-process generation epoch (O(1))."""
    with _GEN_LOCK:
        return _current_generation


def advance_generation() -> int:
    """Increment and return the new generation epoch.

    Called ONLY by the nightly RecallIndex rebuild step after it writes the
    fresh snapshot.  Never called per mutation.  Returns the new epoch.
    """
    global _current_generation  # noqa: PLW0603
    with _GEN_LOCK:
        _current_generation += 1
        return _current_generation


def load_current_generation_from_snapshot(store: Any) -> int:
    """Read the generation epoch from the on-disk snapshot (O(1) decrypt).

    Called at daemon startup so the in-process counter resumes from the
    last persisted epoch rather than starting at 0 after a daemon restart.
    Returns 0 on any read/decrypt failure (safe: triggers epoch-mismatch
    on the first consult until the nightly rebuild stamps a fresh epoch).
    """
    data = _load_and_decrypt_cache(store)
    if data is None:
        return 0
    if data.get("cache_version") != CACHE_VERSION:
        return 0
    gen = data.get("generation", 0)
    try:
        result = int(gen)
        global _current_generation  # noqa: PLW0603
        with _GEN_LOCK:
            if result > _current_generation:
                _current_generation = result
        return result
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------ JSON encode/decode


def _encode_assignment(assignment: Any) -> dict:
    """Serialise CommunityAssignment to a JSON-friendly dict.

    node_to_community and mid_regions have UUID keys; community_centroids
    is {UUID: [float]}. UUIDs are stringified; floats stay native.
    """
    return {
        "node_to_community": {
            str(leaf): str(comm)
            for leaf, comm in getattr(assignment, "node_to_community", {}).items()
        },
        "community_centroids": {
            str(comm): list(vec)
            for comm, vec in getattr(assignment, "community_centroids", {}).items()
        },
        "modularity": float(getattr(assignment, "modularity", 0.0)),
        "backend": str(getattr(assignment, "backend", "flat")),
        "top_communities": [str(c) for c in getattr(assignment, "top_communities", [])],
        "mid_regions": {
            str(comm): [str(m) for m in members]
            for comm, members in getattr(assignment, "mid_regions", {}).items()
        },
    }


def _decode_assignment(raw: dict) -> Any:
    """Inverse of _encode_assignment. Imports CommunityAssignment lazily
    so this module does not pull in the community layer for callers that
    only want to poke the cache file."""
    from iai_mcp.community import CommunityAssignment

    return CommunityAssignment(
        node_to_community={
            UUID(leaf): UUID(comm)
            for leaf, comm in raw.get("node_to_community", {}).items()
        },
        community_centroids={
            UUID(comm): list(vec)
            for comm, vec in raw.get("community_centroids", {}).items()
        },
        modularity=float(raw.get("modularity", 0.0)),
        backend=str(raw.get("backend", "flat")),
        top_communities=[UUID(c) for c in raw.get("top_communities", [])],
        mid_regions={
            UUID(comm): [UUID(m) for m in members]
            for comm, members in raw.get("mid_regions", {}).items()
        },
    )


def _encode_rich_club(rich_club: Any) -> list[str]:
    return [str(u) for u in (rich_club or [])]


def _decode_rich_club(raw: Any) -> list[UUID]:
    return [UUID(u) for u in (raw or [])]


# ----------------------------------------------------------------- size estimator
#
# Bound peak RSS in save() by estimating serialised byte cost without
# materialising the full JSON string.
#
# The legacy save() path encoded the cache payload up to 4 times -- once
# for the initial size check and once after each progressive drop. On
# cold-start graphs (Leiden -> ~1 community per record),
# assignment.community_centroids balloons with len(records) * 384-dim
# float vectors and a single encode call materialises a multi-GB
# intermediate Python string (py-spy profile showed RSS 7.6GB).
#
# The estimator overshoots rather than undershoots: false-positive drops
# are safe (cache stays advisory; cold-start rebuilds from the live store),
# false-negative under-drops produce the very bug we are fixing. The
# constants below are upper bounds for the JSON-encoded byte width of each
# field shape.

# JSON overhead per dict entry: 4 punctuation chars (quotes, colon, comma)
# + variable-length key + value. We track the punctuation explicitly so
# the per-field constants below are pure VALUE budgets.
_JSON_DICT_ENTRY_OVERHEAD: int = 4

# node_payload entry value width upper bound. Shape:
#   {"embedding": [<384 float>], "surface": str(<=256), "centrality": float,
#    "tier": str(<=24), "pinned": bool, "tags": [<=16 short strings],
#    "language": str(<=8)}
# 384-dim float vector dominates: each float worst-case ~24 bytes
# ("-1.2345678901234567,") -> 384*24 = 9216. Plus structural keys / quotes
# ~256. Plus other fields ~512. Round to a comfortable ceiling.
_NODE_PAYLOAD_BYTES_PER_RECORD: int = 10240

# community_centroids entry value width upper bound. Shape:
#   {"<UUID-36>": [<384 float>]}
# 384-dim float same calculus as node_payload embedding -> 9216. Plus
# 36-char UUID quoted -> 38. Plus brackets / commas -> ~16. Round up.
_CENTROID_BYTES_PER_RECORD: int = 9472

# mid_regions entry value width upper bound. Shape:
#   {"<UUID-36>": ["<UUID-36>", ..., "<UUID-36>"]}
# Variable length; bound by typical mid-region size <= 32 UUIDs * 38 bytes
# = 1216, plus brackets / commas -> 1280.
_MID_REGION_BYTES_PER_RECORD: int = 1280

# rich_club is a list of UUID strings: 38 bytes per entry.
_RICH_CLUB_BYTES_PER_ENTRY: int = 38

# Top-level scaffolding (cache_version + key + saved_at + max_degree +
# backend / modularity / top_communities / node_to_community + structural
# JSON braces). Conservative upper bound; node_to_community at scale is
# the variable component.
_BASE_SCAFFOLD_BYTES: int = 4096


def _estimate_serialised_bytes(data: dict) -> int:
    """Upper-bound estimate of the encoded ``data`` dict's byte width
    without actually serialising it.

    Walks the cache payload shape and sums per-field worst-case JSON byte
    widths. Overshoots rather than undershoots so the caller's drop loop
    is conservative (false-positive drops are safe; the cache is advisory
    and cold-start rebuilds from the live store).

    Used by ``save`` before every iteration of the drop loop -- replaces
    the legacy len-of-encoded round-trip which materialised the full
    JSON string up to 4 times per save.
    """
    total = _BASE_SCAFFOLD_BYTES

    # node_payload: dict[str, dict] of per-record graph attributes.
    np_block = data.get("node_payload") or {}
    if isinstance(np_block, dict):
        total += len(np_block) * (
            _NODE_PAYLOAD_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD + 38
        )

    # node_to_community + community_centroids + mid_regions live under
    # data["assignment"]. Encoded shape is what _encode_assignment returns.
    assignment_block = data.get("assignment") or {}
    if isinstance(assignment_block, dict):
        ntc = assignment_block.get("node_to_community") or {}
        if isinstance(ntc, dict):
            # Each entry: "<UUID-36>": <int>; ~50 bytes worst case.
            total += len(ntc) * 50

        centroids = assignment_block.get("community_centroids") or {}
        if isinstance(centroids, dict):
            total += len(centroids) * (
                _CENTROID_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD
            )

        mid = assignment_block.get("mid_regions") or {}
        if isinstance(mid, dict):
            total += len(mid) * (
                _MID_REGION_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD
            )

        top = assignment_block.get("top_communities") or []
        if isinstance(top, list):
            total += len(top) * 16

    rich_club = data.get("rich_club") or []
    if isinstance(rich_club, list):
        total += len(rich_club) * _RICH_CLUB_BYTES_PER_ENTRY

    return total


# ------------------------------------------------------------ public API


def try_load(store: Any) -> tuple | None:
    """Return the cached ``(assignment, rich_club, node_payload, max_degree)``
    tuple if the on-disk file is present, readable, and keyed to the
    current store state. Return ``None`` on any mismatch or error.

    The third element is the ``node_payload`` blob
    (``dict[str, dict]``: UUID-str -> {embedding, surface, centrality,
    tier, pinned}) so cold-start ``build_runtime_graph`` can rehydrate
    graph node attributes without re-walking the encrypted records
    table.

    The fourth element is ``max_degree`` (one int — the maximum graph
    degree in the live graph at save() time). Used by the pipeline rank
    stage to normalise log(1+deg) into [0,1] without re-walking the
    graph. Missing / malformed value coerces to 0 — the rank stage falls
    back to deg_norm=0.0 when max_degree==0 (cosine carries the recall on
    its own at the cold-start scale).

    Callers treat ``None`` as "rebuild from the live graph" — never as
    an error condition. The cache is advisory.

    File format is AES-256-GCM-wrapped JSON. A legacy plaintext file
    (cache_version="06-02-v1") is read once and re-saved under the
    ciphertext format on the same call — one-cycle lazy migration. Any
    decrypt failure (wrong key, tampered file) returns None and the
    caller rebuilds from store.
    """
    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("runtime_graph_cache read failed: %s", exc)
        return None

    legacy_v2_plaintext = False
    if is_encrypted(raw_text):
        # v3 ciphertext path.
        try:
            key = _cache_encryption_key(store)
            plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
            data = json.loads(plaintext_json)
        except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
            # InvalidTag (AES-GCM tag failure: wrong key or tampered file)
            # subclasses only ``Exception``, so it must be named explicitly or
            # it escapes and crashes the recall hot path. The docstring promises
            # "any decrypt failure ... returns None and the caller rebuilds".
            try:
                sys.stderr.write(
                    '{"event":"runtime_graph_cache_decrypt_failed","error":'
                    + json.dumps(str(exc) or type(exc).__name__)
                    + '}\n'
                )
            except (OSError, ValueError):
                pass
            return None
    else:
        # Legacy plaintext path. Accept ONLY the documented v2 cache
        # version; anything else falls through to a clean rebuild
        # (the file is not necessarily ours).
        try:
            data = json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("cache_version") == LEGACY_CACHE_VERSION_PLAINTEXT:
            legacy_v2_plaintext = True
        else:
            # Unknown format / version — treat as no cache.
            return None

    if not isinstance(data, dict):
        return None
    if not legacy_v2_plaintext and data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    current_key = _cache_key(store)
    if legacy_v2_plaintext:
        # Legacy v2 caches embed CACHE_VERSION="06-02-v1" in the last
        # key slot; compare against an expected key that swaps the
        # current CACHE_VERSION for the legacy one. All other
        # invariants (records_count, edges_count, schema_version,
        # embed_dim) MUST still match — anything else means the cache
        # is stale and we rebuild from store.
        expected_legacy_key = tuple(
            list(current_key)[:-1] + [LEGACY_CACHE_VERSION_PLAINTEXT]
        )
        if saved_key != expected_legacy_key:
            return None
    else:
        if saved_key != current_key:
            return None

    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
        node_payload_raw = data.get("node_payload")
        node_payload: dict[str, dict] | None
        if isinstance(node_payload_raw, dict):
            # Shallow dict-of-dicts; embedding list[float] round-trips
            # through JSON natively.
            #
            # Defensively drop poisoned entries on rehydrate. The
            # retrieve.py write-path guards against empty-surface entries,
            # but an existing on-disk cache may still contain them.
            # Belt-and-braces: rehydrate-side
            # filter ensures a poisoned cache from any source (legacy
            # write, future regression, manual tamper) cannot leak an
            # empty/None surface into the live graph.
            #
            # Drop rule: surface in (None, "") OR _decrypt_failed=True.
            # The structured event uses the same stderr-JSON idiom as
            # the existing runtime_graph_cache_decrypt_failed emission
            # at lines 376-383 — runtime_graph_cache.py intentionally
            # bypasses logging because the logger's re-entrant import
            # path can deadlock during cache rehydrate at very-cold-start.
            node_payload = {}
            drop_count = 0
            for k, v in node_payload_raw.items():
                if not isinstance(v, dict):
                    continue
                surface = v.get("surface")
                if surface in (None, "") or v.get("_decrypt_failed"):
                    drop_count += 1
                    continue  # poisoned entry — never expose as a "valid" record
                node_payload[str(k)] = dict(v)
            if drop_count > 0:
                try:
                    sys.stderr.write(
                        '{"event":"runtime_graph_cache_drop_poisoned_entry","count":'
                        + str(drop_count)
                        + '}\n'
                    )
                except OSError:
                    pass
        else:
            node_payload = None
        # max_degree is one int — never participates in the iterative drop
        # path because dropping it costs nothing at the JSON byte-budget level.
        try:
            max_degree = int(data.get("max_degree", 0) or 0)
        except (TypeError, ValueError):
            max_degree = 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache decode failed: %s", exc)
        return None

    if legacy_v2_plaintext:
        # Lazy migration — re-save the loaded content under the current
        # encrypted format. Wrapped: a
        # migration write failure must not block the caller from
        # using the loaded values they already have in memory.
        try:
            save(
                store, assignment, rich_club,
                node_payload=node_payload, max_degree=max_degree,
            )
        except (OSError, ValueError) as exc:
            logger.debug("runtime_graph_cache legacy re-save failed: %s", exc)

    return assignment, rich_club, node_payload, max_degree


def _load_and_decrypt_cache(store: Any) -> "dict | None":
    """Read, decrypt, and JSON-parse the on-disk cache file.

    Returns the raw dict on success; None if the file is absent, unreadable,
    fails the AES-GCM tag check, or is not recognised JSON.  Used by both
    try_load and load_last_good_structural to avoid duplicating the crypto.
    Legacy plaintext (cache_version="06-02-v1") is rejected here — only
    the current AES-GCM ciphertext format is accepted for this helper.
    """
    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        logger.debug("runtime_graph_cache read failed: %s", exc)
        return None
    if not is_encrypted(raw_text):
        return None
    try:
        key = _cache_encryption_key(store)
        plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
        data = json.loads(plaintext_json)
    except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
        # InvalidTag = the AES-GCM auth-tag check failed (wrong key or a
        # corrupt/tampered sidecar). It subclasses only ``Exception`` (NOT any
        # of the others), so without it here a tag failure escapes this helper
        # and crashes the recall pipeline — contradicting the docstring promise
        # to return None "if it fails the AES-GCM tag check". A bad sidecar must
        # degrade to a rebuild, never break recall.
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_decrypt_failed","error":'
                + json.dumps(str(exc) or type(exc).__name__)
                + '}\n'
            )
        except (OSError, ValueError):
            pass
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_last_good_structural(store: Any) -> "tuple | None":
    """Return ``(assignment, rich_club)`` from the on-disk cache, IGNORING
    the records_count/edges_count staleness components of the key.

    Case-2 reader: when try_load returns None because the windowed count
    bucket has crossed a boundary (key-drifted POST-WARM MISS),
    this function decodes the SAME on-disk snapshot but skips only the count
    comparison.  It STILL enforces schema_version + embed_dim + cache_version
    parity, returning None on any of those mismatching or on decrypt failure.

    Safety contract: a cache_version or embed_dim bump → None (the loader
    then takes case-3), NEVER a dimension-mismatched snapshot on the hot path.

    Returns None if:
    - no cache file exists
    - the ciphertext decrypt fails
    - the saved key's schema_version / embed_dim / cache_version differs from
      the current store values
    - the assignment/rich_club decode fails

    Returns None also when try_load would succeed (key fully matches) — callers
    should call try_load first and only fall back to this function on a MISS.
    """
    data = _load_and_decrypt_cache(store)
    if data is None:
        return None
    # Reject caches not on the current format.
    if data.get("cache_version") != CACHE_VERSION:
        return None
    # Extract the saved key.
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 5:
        return None
    # Count-free parity: schema_version (index 2), embed_dim (index 3),
    # cache_version (index 4).  Skip count buckets (indices 0, 1).
    # Uses _parity_components so this function is safe on the recall hot path
    # with no active_records_count / count_rows calls.
    current_parity = _parity_components(store)
    if saved_key[2] != current_parity[0]:  # schema_version
        return None
    if saved_key[3] != current_parity[1]:  # embed_dim
        return None
    if saved_key[4] != current_parity[2]:  # cache_version
        return None
    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache last_good decode failed: %s", exc)
        return None
    return assignment, rich_club


def load_recall_structural(store: Any) -> "tuple":
    """4-case consume-only recall-context loader.

    Returns ``(assignment, rich_club, max_degree, structural_source)`` where
    ``structural_source`` is one of "overlay", "normal", "last_good", or
    "cold_degrade" and ``max_degree`` is the global max-degree int from the
    persisted cache (0 on cold-degrade or when the cache predates the field).

    CASE 0 — OVERLAY HIT (Layer-2): consult_overlay returns (assignment, rich_club)
    when the snapshot's generation epoch matches the in-process epoch, the O(1)
    freshness fuse has NOT tripped (max_age + in-process dirty counter both within
    threshold), and snapshot-internal invariants pass.  structural_source = "overlay".
    No count(*) / active_records_count / count_rows on the hot path.
    On any overlay bypass the function falls through to CASE 1/2/3.

    CASE 1 — KEY-MATCHED HIT: try_load returns the cached assignment + rich_club
    (single-write HIT via the windowed key).  structural_source = "normal".

    CASE 2 — KEY-DRIFTED POST-WARM MISS: try_load returns None but the cache
    file exists and parity checks pass → load_last_good_structural returns the
    last-good GLOBAL assignment + rich_club.  structural_source = "last_good".
    A slightly-stale global rich-club is far better than empty (which would
    drop hub-sensitive gold).

    CASE 3 — TRULY COLD: no cache file (or parity mismatch that forced
    load_last_good_structural to return None too) → empty assignment + empty
    rich_club + max_degree=0 + structural_source = "cold_degrade".  The caller
    should stamp _source="cold-structural-degrade" on the response so the loss
    is observable.

    This function NEVER calls build_runtime_graph / detect_communities /
    rich_club_nodes — it is consume-only, daemon-independent, and safe to
    call on the recall hot path.
    """
    from iai_mcp.community import CommunityAssignment  # lazy import, avoid circular

    # CASE 0 — Layer-2 overlay O(1) HIT (consult_overlay is count-free).
    # On any bypass, fall through to the Layer-1 cases below.
    # Lazy-init: if the in-process generation counter is 0 (fresh daemon or
    # first recall after restart) and a snapshot exists on disk, load its
    # stamped generation so the epoch comparison is valid without waiting for
    # the next nightly rebuild.  O(1) decrypt on first call only.
    if get_current_generation() == 0:
        load_current_generation_from_snapshot(store)
    try:
        overlay_result = consult_overlay(store)
        if not isinstance(overlay_result, _OverlayBypass):
            ov_assignment, ov_rich_club = overlay_result
            # max_degree: read from snapshot.
            data = _load_and_decrypt_cache(store)
            ov_max_degree = 0
            if data is not None:
                try:
                    ov_max_degree = int(data.get("max_degree", 0) or 0)
                except (TypeError, ValueError):
                    ov_max_degree = 0
            return ov_assignment, ov_rich_club, ov_max_degree, "overlay"
    except Exception:  # noqa: BLE001 -- overlay errors must never break recall
        pass

    cached = try_load(store)
    if cached is not None:
        assignment, rich_club, _node_payload, max_degree = cached
        return assignment, rich_club, int(max_degree or 0), "normal"

    last_good = load_last_good_structural(store)
    if last_good is not None:
        assignment, rich_club = last_good
        # max_degree is not returned by load_last_good_structural; use 0 so
        # the bounded assembler falls back to graph-local degrees (still
        # correct; slightly less accurate than the cached global value).
        return assignment, rich_club, 0, "last_good"

    # Truly cold: return empty structural bias.
    empty_assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="cold-degrade",
        top_communities=[],
        mid_regions={},
    )
    return empty_assignment, [], 0, "cold_degrade"


# Module-level override for the rebuild_timestamp written by save_with_generation.
# Set immediately before calling save(), cleared immediately after.  Protected
# by _GEN_LOCK for thread safety (the nightly step is single-threaded, but
# defensive).
_rebuild_timestamp_override: str = ""


def save(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
) -> bool:
    """Persist the cache atomically. Returns True on success, False on
    any write error. Errors are swallowed — the caller has freshly
    computed values in memory either way; a failed cache write is not
    a reason to break the recall path.

    ``node_payload`` persists the per-record graph-node attribute map
    (UUID-str -> {embedding: list[float], surface: str, centrality: float,
    tier: str, pinned: bool}). Absent / None -> the cache still writes
    assignment + rich_club and next cold-start will rebuild node payload
    from the live store walk. JSON-native shape (no binary serialisation)
    keeps the cache auditable.

    ``max_degree`` (one int) is the maximum graph degree at save() time.
    Used by the rank stage to normalise log(1+deg) into [0,1] without
    re-walking the graph on every recall. Always present in the payload —
    never participates in the iterative drop path (one int costs nothing
    against MAX_CACHE_BYTES).
    """
    path = _cache_path(store)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    # Normalise node_payload for JSON: stringify keys, list() embeddings.
    encoded_node_payload: dict[str, dict] | None = None
    if node_payload:
        encoded_node_payload = {}
        for k, v in node_payload.items():
            if not isinstance(v, dict):
                continue
            # Embeddings can be numpy float32 from store rows; coerce to
            # plain Python float so json.dump does not trip on
            # "Object of type float32 is not JSON serializable".
            raw_emb = v.get("embedding") or []
            # `centrality` is betweenness, computed once during
            # build_runtime_graph and persisted so warm starts don't
            # recompute it. Missing/None coerces to 0.0. `tags`/`language`
            # persisted so SimpleRecordView surfaces the full
            # profile_modulation input set without a store.get fallback.
            raw_tags = v.get("tags") or []
            encoded_node_payload[str(k)] = {
                "embedding": [float(x) for x in raw_emb],
                "surface": str(v.get("surface", "")),
                "centrality": float(v.get("centrality") or 0.0),
                "tier": str(v.get("tier", "episodic")),
                "pinned": bool(v.get("pinned", False)),
                "tags": [str(t) for t in raw_tags if t is not None],
                "language": str(v.get("language", "en") or "en"),
            }

    data = {
        "cache_version": CACHE_VERSION,
        "key": list(_cache_key(store)),
        "assignment": _encode_assignment(assignment),
        "rich_club": _encode_rich_club(rich_club),
        "node_payload": encoded_node_payload or {},
        # max_degree is one int — survives every iterative drop step below
        # because dropping it saves no measurable bytes.
        "max_degree": int(max_degree or 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        # Layer-2 overlay: generation epoch (nightly-stamp only) +
        # rebuild_timestamp (wall-clock baseline for the O(1) freshness fuse).
        # Written by save_with_generation() from the nightly rebuild step;
        # absent/0 in snapshots written by build_runtime_graph (non-nightly
        # paths produce an epoch=0 snapshot that the overlay consult will
        # reject as epoch_mismatch until the first nightly rebuild).
        "generation": int(get_current_generation()),
        "rebuild_timestamp": _rebuild_timestamp_override or "",
    }

    # Size guard: the previous single-drop path only trimmed
    # ``node_payload`` and shipped whatever remained, even when the bloat
    # lived elsewhere. On an all-isolated graph (0 edges) Leiden returns
    # one community per node and ``assignment.community_centroids`` alone
    # balloons to 70+ MiB (one 384-dim float vector per record).
    #
    # Drop candidates in decreasing marginal-value order. Estimate the
    # encoded byte cost BEFORE materialising
    # the JSON string, so peak RSS during save matches the final on-disk
    # file size instead of the pre-drop full payload size. ``json.dumps``
    # is called AT MOST ONCE per ``save`` invocation, after all drop
    # decisions are made. The authoritative slim output of Leiden
    # (``node_to_community``, ``top_communities``, ``modularity``,
    # ``backend``) and the ``rich_club`` list always survive -- they are
    # cheap to encode and expensive to recompute from the live store.
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        # 1) node_payload: per-record blob, rebuildable from the live
        #    store walk on the next cold start.
        data["node_payload"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        # 2) assignment.community_centroids: {UUID: [float; embed_dim]}.
        #    On sparse graphs this is the biggest single field. Leiden
        #    recomputes centroids on the next build.
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["community_centroids"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        # 3) assignment.mid_regions: {UUID: [UUID, ...]}. Smaller view;
        #    also recomputable.
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["mid_regions"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        # Still over the cap after dropping every advisory field. Prefer
        # a clean "give up" to shipping an oversized file; the caller
        # already has the in-memory values and the next build will
        # recompute everything from the live store.
        return False

    # Single final encode -- AT MOST ONE json.dumps per save() call.
    serialised = json.dumps(data, ensure_ascii=False)

    # Encrypt the JSON payload before writing. Same AES-256-GCM machinery
    # + key as the literal_surface column. ASCII-only ciphertext
    # (b64 envelope) lets us keep the text-mode write path; on-disk
    # plaintext canary is provably absent.
    try:
        key = _cache_encryption_key(store)
        ciphertext = encrypt_field(serialised, key, _CACHE_AAD)
    except (OSError, ValueError, RuntimeError) as exc:
        # Encryption failure: skip the cache write rather than persist
        # plaintext on disk. Cache is advisory; recall path unaffected.
        logger.debug("runtime_graph_cache encrypt failed: %s", exc)
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_encrypt_failed"}\n'
            )
        except OSError:
            pass
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-snapshot durability: flush Python's stdio buffer and
        # fsync the kernel buffer to stable storage BEFORE the with-
        # block closes the file. POSIX guarantees os.replace is
        # atomic; this fsync guarantees the source inode is durable
        # so a mid-write crash leaves either the old snapshot or the
        # new snapshot — never a truncated file. Parent-directory
        # fsync is intentionally skipped: APFS journals directory
        # entries and ext4 with default mount handles the rename via
        # its journal; the macOS + Linux defaults the project ships
        # on do not need it.
        with tmp_path.open("w", encoding="ascii") as f:
            f.write(ciphertext)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
        return True
    except OSError as exc:
        logger.debug("runtime_graph_cache write failed: %s", exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def save_with_generation(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
) -> bool:
    """Nightly-rebuild variant of save() that advances the generation epoch
    and stamps the rebuild_timestamp into the snapshot.

    Called ONLY by the nightly RecallIndex rebuild sleep step (sleep_pipeline.py).
    Never called per mutation — that would self-invalidate the overlay on
    recall-path boost_edges writes.

    Steps performed atomically (best-effort):
    1. Advance the in-process generation epoch.
    2. Reset the in-process record-mutation dirty counter to zero.
    3. Write the snapshot with the new generation + rebuild_timestamp.

    Returns True if the snapshot was persisted successfully, False on any
    write/encrypt error (the pipeline step should log but not raise on False —
    the overlay is advisory; recall still works via Layer-1).
    """
    new_gen = advance_generation()
    reset_dirty_counter()
    # Build snapshot via the normal save() logic but override the generation
    # and rebuild_timestamp fields.  We do this by calling save() which now
    # writes get_current_generation() (which IS new_gen) into the payload,
    # then we need to also stamp the rebuild_timestamp.
    #
    # Implementation: write the snapshot by temporarily patching data before
    # the encrypt step. To avoid duplicating save()'s ~100 LOC drop logic,
    # we write via a two-step approach: save() always writes
    # get_current_generation() (we just advanced it) and rebuild_timestamp=""
    # into the payload, then we post-patch the rebuild_timestamp by reading
    # back the plaintext, updating, and re-encrypting.  But that races.
    #
    # Simpler: inline the rebuild_timestamp into the snapshot at save() time
    # by using a module-level "pending rebuild timestamp" flag that save()
    # reads and then clears.  Still racy in theory but in practice the nightly
    # step is single-threaded.
    #
    # Cleanest: patch the data dict AFTER save() returns by re-reading and
    # re-writing.  But that risks losing the size-budget drops.
    #
    # Chosen: rebuild_timestamp is written by save() reading a module-level
    # _rebuild_timestamp_override when non-empty, then clearing it.
    ts_iso = datetime.now(timezone.utc).isoformat()
    global _rebuild_timestamp_override  # noqa: PLW0603
    with _GEN_LOCK:
        _rebuild_timestamp_override = ts_iso
    result = save(store, assignment, rich_club, node_payload=node_payload, max_degree=max_degree)
    with _GEN_LOCK:
        _rebuild_timestamp_override = ""
    return result


def invalidate(store: Any) -> None:
    """Delete the cache file for ``store``. Safe when the file does not
    exist. Used by explicit ``needs_refresh`` signals and by tests that
    want a clean slate."""
    path = _cache_path(store)
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.debug("runtime_graph_cache invalidate failed: %s", exc)


def _rebuild_and_save_rgc(store: Any) -> dict:
    """Build a fresh structural snapshot from SQLite ground truth and persist it.

    Builds a MemoryGraph from ``store.all_records()`` and the edges table,
    runs ``detect_communities`` + ``rich_club_nodes``, computes
    ``max_degree``, and calls ``save_with_generation``.

    Returns a result dict with keys:
      - ``rebuilt`` (bool)  — True when the rebuild body completed
      - ``saved`` (bool)    — True when ``save_with_generation`` returned True
      - ``node_count`` (int)
      - ``generation`` (int)

    Raises on hard failure; the caller (daemon trigger or test) decides how
    to handle the exception.

    Thread-safety notes:
    - The store reads (``store.all_records()`` and the edges-table
      ``to_pandas()``) internally acquire the Hippo shared-connection
      ``_conn_lock`` — the same discipline followed by the nightly sleep
      step whose body this helper factors out.  No additional wrapping is
      needed here; callers must not hold ``_conn_lock`` when calling this
      function (avoid double-acquisition on the re-entrant RLock).
    - The cache-file write via ``save_with_generation`` is atomic
      (tempfile + ``os.replace``) and separate from the store reads.

    This helper is intentionally unconditional (no interrupt check).
    Interrupt-gating, if required, stays in the sleep-step wrapper.
    The ``rebuild_ready`` Event is NOT set here — the daemon caller owns
    the Event so this helper remains reusable by the sleep step without
    side-effects on the wake-path flag.
    """
    from iai_mcp.community import detect_communities
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.richclub import rich_club_nodes

    graph = MemoryGraph()

    try:
        all_records = list(store.all_records())
    except Exception as exc:  # noqa: BLE001
        logger.warning("_rebuild_and_save_rgc: all_records failed: %s", exc)
        raise

    for rec in all_records:
        try:
            graph.add_node(
                rec.id,
                community_id=None,
                embedding=list(rec.embedding),
            )
            graph.set_node_payload(rec.id, {
                "embedding": list(rec.embedding),
                "surface": rec.literal_surface,
                "centrality": float(rec.centrality),
                "tier": rec.tier,
                "pinned": bool(rec.pinned),
                "tags": list(getattr(rec, "tags", []) or []),
                "language": str(getattr(rec, "language", "en") or "en"),
            })
        except Exception:  # noqa: BLE001
            pass

    try:
        edges_df = store.db.open_table("edges").to_pandas()
        if not edges_df.empty:
            from uuid import UUID as _UUID
            for _, row in edges_df.iterrows():
                try:
                    src = _UUID(str(row["src"]))
                    dst = _UUID(str(row["dst"]))
                    w = float(row.get("weight", 1.0) or 1.0)
                    graph.add_edge(src, dst, weight=w)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("_rebuild_and_save_rgc: edge load failed: %s", exc)

    assignment = detect_communities(graph, prior_mode="cold")
    rc = rich_club_nodes(graph)

    max_degree = 0
    try:
        for _nid, deg in graph.degrees():
            if deg > max_degree:
                max_degree = deg
    except Exception:  # noqa: BLE001
        pass

    saved = save_with_generation(store, assignment, rc, max_degree=max_degree)

    node_count = graph.node_count()
    return {
        "rebuilt": True,
        "saved": saved,
        "node_count": int(node_count),
        "generation": get_current_generation(),
    }
