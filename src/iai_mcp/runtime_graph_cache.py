"""Plan 05-09 (P4.A): persist Leiden community assignment + rich-club
to disk so the first ``memory_recall`` call in a fresh core process
does not rebuild these expensive artefacts from scratch.

The Phase-1 ``retrieve.build_runtime_graph`` rebuilds everything on
every call:

    graph   = MemoryGraph()          # ~100 ms to construct from rows
    detect_communities(graph)        # Leiden, ~200 ms at N=1k
    rich_club_nodes(graph, 0.10)     # ~20 ms

Phase-5 P4 measured first-call cold path at ~440 ms at N=1k. Caching
the *Leiden output* and the rich-club node list eliminates the two
expensive computations when the store has not changed. MemoryGraph
construction itself is cheap enough to rebuild per call; caching it
too would require pickle (the NetworkX graph is not JSON-friendly)
and the security-vs-speed trade-off is not worth it for ~100 ms.

**Invalidation** — any of these triggers a rebuild:

- Record count changed (user saved / consolidated / merged)
- Edge count changed (Hebbian reinforcement or contradiction added)
- SCHEMA_VERSION_CURRENT bumped (store migrated)
- store.embed_dim changed (user swapped embedder; Plan 05-08)
- CACHE_VERSION bumped (this module's on-disk format changed)

Any inconsistency — corrupt JSON, unreadable file, unknown keys —
falls through to a clean rebuild. The cache is purely an optimisation;
the authoritative graph is always the LanceDB store.

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

Constitutional invariants:

- C3 (zero API): pure local JSON + filesystem operations.
- C6 (read-only against store): cache writes go to the cache file
  only, never to any LanceDB table.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from iai_mcp.crypto import (
    CryptoKey,
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import SCHEMA_VERSION_CURRENT


# Bump this whenever the on-disk cache shape changes. A mismatch
# forces every user on the old shape to rebuild -- safer than silently
# loading a file whose key contract has drifted.
#
# R2: bumped to "06-02-v1" — payload now carries max_degree
# (one int) so the rank stage can normalise log(1+deg) by log(1+max_deg)
# without re-walking the live graph on every recall. Old caches lacking
# the field are invalidated cleanly by the version bump and rebuild on
# the next build_runtime_graph call.
#
# W3 / bumped to "07-09-v3" — cache file is now
# AES-256-GCM-wrapped. Old "06-02-v1" caches that pre-date 07.9 are
# treated as legacy plaintext: read once, lazily re-saved as ciphertext
# on first warm-start under 07.9, then never read again.
CACHE_VERSION: str = "07-09-v3"
LEGACY_CACHE_VERSION_PLAINTEXT: str = "06-02-v1"

# AES-GCM associated data (AD): binds the ciphertext to this format and
# version. A bytewise tampering attempt that swaps the file with a
# v06-02-v1 plaintext or any other stream fails the decrypt tag check.
_CACHE_AAD: bytes = b"runtime-graph-cache:v3"

CACHE_FILENAME: str = "runtime_graph_cache.json"

# Size cap for the on-disk cache. When the encoded payload exceeds this,
# ``save`` drops ``node_payload`` (the large per-record embedding map) and
# writes only ``assignment + rich_club``. Cold-start ``build_runtime_graph``
# rehydrates the node payload from the LanceDB store on the next recall;
# the cache remains advisory. 10 MiB holds the Leiden + rich-club artefacts
# for a ~50k-record store comfortably while keeping cold-start load under
# the session-start token budget.
MAX_CACHE_BYTES: int = 10 * 1024 * 1024


def _cache_path(store: Any) -> Path:
    """Cache file lives next to the LanceDB directory so it travels with
    the store on backup / move. One cache file per MemoryStore."""
    root = getattr(store, "root", None)
    if root is None:
        root = Path.cwd()
    return Path(root) / CACHE_FILENAME


def _cache_encryption_key(store: Any) -> bytes:
    """Phase 07.9 W3 / 32-byte AES key for the runtime-graph-cache
    sidecar. Reuses the store's already-cached key whenever possible to
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
        except Exception:
            pass
    user_id = getattr(store, "user_id", "default") or "default"
    return CryptoKey(user_id=user_id).get_or_create()


def _cache_key(store: Any) -> tuple:
    """Monotonic identity for "the cached graph is still correct for this
    store state". Any change to a component invalidates the cache.

    (records_count, edges_count, schema_version, embed_dim, cache_version)
    """
    try:
        records_count = int(store.db.open_table("records").count_rows())
    except Exception:
        records_count = -1
    try:
        edges_count = int(store.db.open_table("edges").count_rows())
    except Exception:
        edges_count = -1
    embed_dim = int(getattr(store, "embed_dim", 0))
    return (
        records_count,
        edges_count,
        SCHEMA_VERSION_CURRENT,
        embed_dim,
        CACHE_VERSION,
    )


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
# W2 / D-07, D-08, bound peak RSS in save() by estimating
# serialised byte cost without materialising the full JSON string.
#
# The legacy save() path encoded the cache payload up to 4 times -- once
# for the initial size check and once after each progressive drop. On
# cold-start graphs (Leiden -> ~1 community per record),
# assignment.community_centroids balloons with len(records) * 384-dim
# float vectors and a single encode call materialises a multi-GB
# intermediate Python string (py-spy confirmed RSS 7.6GB on cold start).
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

    the third element is the ``node_payload`` blob
    (``dict[str, dict]``: UUID-str -> {embedding, surface, centrality,
    tier, pinned}) so cold-start ``build_runtime_graph`` can rehydrate
    NetworkX node attributes without re-walking the encrypted records
    table.

    R2: the fourth element is ``max_degree`` (one int — the
    maximum NetworkX degree in the live graph at save() time). Used by
    the pipeline rank stage to normalise log(1+deg) into [0,1] without
    re-walking the graph. Missing / malformed value coerces to 0 — the
    rank stage falls back to deg_norm=0.0 when max_degree==0 (cosine
    carries the recall on its own at the cold-start scale).

    Callers treat ``None`` as "rebuild from the live graph" — never as
    an error condition. The cache is advisory.

    W3 / file format is now AES-256-GCM-wrapped JSON.
    A pre-07.9 plaintext file (cache_version="06-02-v1") is read once
    and re-saved under the new ciphertext format on the same call —
    one-cycle lazy migration. Any decrypt failure (wrong key, tampered
    file) returns None and the caller rebuilds from store.
    """
    path = _cache_path(store)
    if not path.exists():
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    legacy_v2_plaintext = False
    if is_encrypted(raw_text):
        # v3 ciphertext path.
        try:
            key = _cache_encryption_key(store)
            plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
            data = json.loads(plaintext_json)
        except Exception as exc:
            try:
                sys.stderr.write(
                    '{"event":"runtime_graph_cache_decrypt_failed","error":'
                    + json.dumps(str(exc))
                    + '}\n'
                )
            except Exception:
                pass
            return None
    else:
        # Legacy plaintext path. Accept ONLY the documented v2 cache
        # version; anything else falls through to a clean rebuild
        # (the file is not necessarily ours).
        try:
            data = json.loads(raw_text)
        except Exception:
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
            # Plan 07.11-02 / (V2-03 fix): defensively drop
            # poisoned entries on rehydrate. Even though Plan 07.11-02's
            # retrieve.py fix prevents future writes of empty-surface
            # entries, an existing on-disk cache from before this fix
            # may still contain them. Belt-and-braces: rehydrate-side
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
                except Exception:
                    pass
        else:
            node_payload = None
        # R2: max_degree is one int — never participates in
        # the iterative drop path because dropping it costs nothing at
        # the JSON byte-budget level.
        try:
            max_degree = int(data.get("max_degree", 0) or 0)
        except (TypeError, ValueError):
            max_degree = 0
    except Exception:
        return None

    if legacy_v2_plaintext:
        # W3 / lazy migration — re-save the loaded
        # content under the new v3 encrypted format. Wrapped: a
        # migration write failure must not block the caller from
        # using the loaded values they already have in memory.
        try:
            save(
                store, assignment, rich_club,
                node_payload=node_payload, max_degree=max_degree,
            )
        except Exception:
            pass

    return assignment, rich_club, node_payload, max_degree


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

    ``node_payload`` persists the per-record graph-node
    attribute map (UUID-str -> {embedding: list[float], surface: str,
    centrality: float, tier: str, pinned: bool}). Absent / None -> the
    cache still writes assignment + rich_club and next cold-start will
    rebuild node payload from the live store walk. JSON-native shape
    (no binary serialisation) keeps the cache auditable.

    R2: ``max_degree`` (one int) is the maximum graph degree
    at save() time. Used by the rank stage to normalise log(1+deg) into
    [0,1] without re-walking the graph on every recall. Always present
    in the payload — never participates in the iterative drop path
    (one int costs nothing against MAX_CACHE_BYTES).
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
            # embeddings can be numpy float32 from LanceDB
            # rows; coerce to plain Python float so json.dump does not
            # trip on "Object of type float32 is not JSON serializable".
            raw_emb = v.get("embedding") or []
            # `centrality` is now betweenness, computed once
            # during build_runtime_graph and persisted here so warm starts
            # don't recompute it. Missing/None coerces to 0.0 (legacy
            # pre-05-13 pre-compute shape). `tags`/`language` persisted
            # so SimpleRecordView surfaces the full profile_modulation
            # input set without a store.get fallback.
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
        # R2: max_degree is one int — survives every iterative
        # drop step below because dropping it saves no measurable bytes.
        "max_degree": int(max_degree or 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    # Size guard: the previous single-drop path only trimmed
    # ``node_payload`` and shipped whatever remained, even when the bloat
    # lived elsewhere. On an all-isolated graph (0 edges) Leiden returns
    # one community per node and ``assignment.community_centroids`` alone
    # balloons to 70+ MiB (one 384-dim float vector per record).
    #
    # Drop candidates in decreasing marginal-value order. W2 /
    # D-07, D-08, estimate the encoded byte cost BEFORE materialising
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

    # Single final encode -- AT MOST ONE json.dumps per save() per D-10.
    serialised = json.dumps(data, ensure_ascii=False)

    # W3 / encrypt the JSON payload before writing.
    # Same AES-256-GCM machinery + key as the LanceDB literal_surface
    # column. ASCII-only ciphertext (b64 envelope) lets us keep the
    # text-mode write path; on-disk plaintext canary is provably absent.
    try:
        key = _cache_encryption_key(store)
        ciphertext = encrypt_field(serialised, key, _CACHE_AAD)
    except Exception:
        # Encryption failure: skip the cache write rather than persist
        # plaintext on disk. Cache is advisory; recall path unaffected.
        try:
            sys.stderr.write(
                '{"event":"runtime_graph_cache_encrypt_failed"}\n'
            )
        except Exception:
            pass
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="ascii") as f:
            f.write(ciphertext)
        os.replace(str(tmp_path), str(path))
        return True
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


def invalidate(store: Any) -> None:
    """Delete the cache file for ``store``. Safe when the file does not
    exist. Used by explicit ``needs_refresh`` signals and by tests that
    want a clean slate."""
    path = _cache_path(store)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
