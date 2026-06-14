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

preload_ready: threading.Event = threading.Event()

rebuild_ready: threading.Event = threading.Event()


CACHE_VERSION: str = "62-02-v5"

_STALENESS_WINDOW: int = 10
LEGACY_CACHE_VERSION_PLAINTEXT: str = "06-02-v1"

_CACHE_AAD: bytes = b"runtime-graph-cache:v3"

CACHE_FILENAME: str = "runtime_graph_cache.json"


_FUSE_MAX_AGE_SECONDS: float = 25.0 * 3600.0

_FUSE_DIRTY_THRESHOLD: int = 50

_dirty_counter: int = 0
_DIRTY_COUNTER_LOCK = threading.Lock()


def increment_dirty_counter() -> None:
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter += 1


def reset_dirty_counter() -> None:
    global _dirty_counter  # noqa: PLW0603
    with _DIRTY_COUNTER_LOCK:
        _dirty_counter = 0


def get_dirty_counter() -> int:
    with _DIRTY_COUNTER_LOCK:
        return _dirty_counter


# One shared graph instance reused across refreshes so the allocator footprint
# stays bounded (a fresh instance per cycle fragments the heap arenas). The lock
# serializes concurrent refreshes so adjacency cannot be corrupted mid-rebuild.
_persistent_graph = None
_PERSISTENT_GRAPH_LOCK = threading.Lock()


def _get_persistent_graph():
    global _persistent_graph  # noqa: PLW0603
    if _persistent_graph is None:
        from iai_mcp.graph import MemoryGraph
        _persistent_graph = MemoryGraph()
    return _persistent_graph


MAX_CACHE_BYTES: int = 10 * 1024 * 1024


def _cache_path(store: Any) -> Path:
    root = getattr(store, "root", None)
    if root is None:
        root = Path.cwd()
    return Path(root) / CACHE_FILENAME


def _cache_encryption_key(store: Any) -> bytes:
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
    embed_dim = int(getattr(store, "embed_dim", 0))
    return (SCHEMA_VERSION_CURRENT, embed_dim, CACHE_VERSION)


class _OverlayBypass:
    __slots__ = ("reason", "age_ms")

    def __init__(self, reason: str, age_ms: int = 0) -> None:
        self.reason = reason
        self.age_ms = age_ms

    def __repr__(self) -> str:  # pragma: no cover
        return f"_OverlayBypass(reason={self.reason!r}, age_ms={self.age_ms})"


def _check_snapshot_invariants(data: dict) -> bool:
    assignment_raw = data.get("assignment")
    if not isinstance(assignment_raw, dict):
        return False
    node_to_community = assignment_raw.get("node_to_community") or {}
    if not isinstance(node_to_community, dict):
        return False
    n_communities = len(set(node_to_community.values()))
    if n_communities == 0 and len(node_to_community) > 0:
        return False
    if n_communities > 100_000:
        return False
    rich_club_raw = data.get("rich_club") or []
    if isinstance(rich_club_raw, list) and rich_club_raw:
        node_ids = set(node_to_community.keys())
        for rc_id in rich_club_raw:
            if rc_id not in node_ids:
                return False
    try:
        modularity = float(assignment_raw.get("modularity", 0.0) or 0.0)
        if not (-1.0 <= modularity <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


def consult_overlay(store: Any) -> "tuple | _OverlayBypass":
    data = _load_and_decrypt_cache(store)
    if data is None:
        return _OverlayBypass("no_snapshot")

    if data.get("cache_version") != CACHE_VERSION:
        return _OverlayBypass("parity_mismatch")

    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 5:
        return _OverlayBypass("parity_mismatch")
    current_parity = _parity_components(store)
    if saved_key[2] != current_parity[0]:
        return _OverlayBypass("parity_mismatch")
    if saved_key[3] != current_parity[1]:
        return _OverlayBypass("parity_mismatch")
    if saved_key[4] != current_parity[2]:
        return _OverlayBypass("parity_mismatch")

    snapshot_generation = data.get("generation", 0)
    if not isinstance(snapshot_generation, int):
        return _OverlayBypass("epoch_mismatch")
    current_gen = get_current_generation()
    if current_gen == 0 or snapshot_generation == 0 or snapshot_generation != current_gen:
        return _OverlayBypass("epoch_mismatch")

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
            age_sec = _FUSE_MAX_AGE_SECONDS + 1.0
            age_ms = int(age_sec * 1000)
    else:
        age_sec = 0.0
        age_ms = 0

    dirty = get_dirty_counter()
    if age_sec > _FUSE_MAX_AGE_SECONDS or dirty > _FUSE_DIRTY_THRESHOLD:
        _emit_freshness_fuse_tripped(store, age_ms=age_ms)
        return _OverlayBypass("fuse_tripped", age_ms=age_ms)

    if not _check_snapshot_invariants(data):
        return _OverlayBypass("invariant_failure")

    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache overlay decode failed: %s", exc)
        return _OverlayBypass("invariant_failure")

    return assignment, rich_club


def _emit_freshness_fuse_tripped(store: Any, *, age_ms: int) -> None:
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


_current_generation: int = 0
_GEN_LOCK = threading.Lock()


def get_current_generation() -> int:
    with _GEN_LOCK:
        return _current_generation


def advance_generation() -> int:
    global _current_generation  # noqa: PLW0603
    with _GEN_LOCK:
        _current_generation += 1
        return _current_generation


def load_current_generation_from_snapshot(store: Any) -> int:
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


def _encode_assignment(assignment: Any) -> dict:
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


_JSON_DICT_ENTRY_OVERHEAD: int = 4
# 384-dim float vector dominates: 384*24=9216 + structural ~1024
_NODE_PAYLOAD_BYTES_PER_RECORD: int = 10240
# 384-dim float same calculus as node_payload embedding -> 9216 + UUID
_CENTROID_BYTES_PER_RECORD: int = 9472

_MID_REGION_BYTES_PER_RECORD: int = 1280

_RICH_CLUB_BYTES_PER_ENTRY: int = 38

_BASE_SCAFFOLD_BYTES: int = 4096


def _estimate_serialised_bytes(data: dict) -> int:
    total = _BASE_SCAFFOLD_BYTES

    np_block = data.get("node_payload") or {}
    if isinstance(np_block, dict):
        total += len(np_block) * (
            _NODE_PAYLOAD_BYTES_PER_RECORD + _JSON_DICT_ENTRY_OVERHEAD + 38
        )

    assignment_block = data.get("assignment") or {}
    if isinstance(assignment_block, dict):
        ntc = assignment_block.get("node_to_community") or {}
        if isinstance(ntc, dict):
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


def try_load(store: Any) -> tuple | None:
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
        try:
            key = _cache_encryption_key(store)
            plaintext_json = decrypt_field(raw_text, key, _CACHE_AAD)
            data = json.loads(plaintext_json)
        except (InvalidTag, OSError, ValueError, KeyError, RuntimeError) as exc:
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
        try:
            data = json.loads(raw_text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("cache_version") == LEGACY_CACHE_VERSION_PLAINTEXT:
            legacy_v2_plaintext = True
        else:
            return None

    if not isinstance(data, dict):
        return None
    if not legacy_v2_plaintext and data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    current_key = _cache_key(store)
    if legacy_v2_plaintext:
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
            node_payload = {}
            drop_count = 0
            for k, v in node_payload_raw.items():
                if not isinstance(v, dict):
                    continue
                surface = v.get("surface")
                if surface in (None, "") or v.get("_decrypt_failed"):
                    drop_count += 1
                    continue
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
        try:
            max_degree = int(data.get("max_degree", 0) or 0)
        except (TypeError, ValueError):
            max_degree = 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache decode failed: %s", exc)
        return None

    if legacy_v2_plaintext:
        try:
            save(
                store, assignment, rich_club,
                node_payload=node_payload, max_degree=max_degree,
            )
        except (OSError, ValueError) as exc:
            logger.debug("runtime_graph_cache legacy re-save failed: %s", exc)

    return assignment, rich_club, node_payload, max_degree


def _load_and_decrypt_cache(store: Any) -> "dict | None":
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
    data = _load_and_decrypt_cache(store)
    if data is None:
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    saved_key = tuple(data.get("key", []))
    if len(saved_key) < 5:
        return None
    current_parity = _parity_components(store)
    if saved_key[2] != current_parity[0]:
        return None
    if saved_key[3] != current_parity[1]:
        return None
    if saved_key[4] != current_parity[2]:
        return None
    try:
        assignment = _decode_assignment(data["assignment"])
        rich_club = _decode_rich_club(data.get("rich_club"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("runtime_graph_cache last_good decode failed: %s", exc)
        return None
    return assignment, rich_club


def load_recall_structural(store: Any) -> "tuple":
    from iai_mcp.community import CommunityAssignment

    if get_current_generation() == 0:
        load_current_generation_from_snapshot(store)
    try:
        overlay_result = consult_overlay(store)
        if not isinstance(overlay_result, _OverlayBypass):
            ov_assignment, ov_rich_club = overlay_result
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
        return assignment, rich_club, 0, "last_good"

    empty_assignment = CommunityAssignment(
        node_to_community={},
        community_centroids={},
        modularity=0.0,
        backend="cold-degrade",
        top_communities=[],
        mid_regions={},
    )
    return empty_assignment, [], 0, "cold_degrade"


_rebuild_timestamp_override: str = ""


def save(
    store: Any,
    assignment: Any,
    rich_club: Any,
    node_payload: "dict[str, dict] | None" = None,
    max_degree: int = 0,
) -> bool:
    path = _cache_path(store)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    encoded_node_payload: dict[str, dict] | None = None
    if node_payload:
        encoded_node_payload = {}
        for k, v in node_payload.items():
            if not isinstance(v, dict):
                continue
            raw_emb = v.get("embedding") or []
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
        "max_degree": int(max_degree or 0),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": int(get_current_generation()),
        "rebuild_timestamp": _rebuild_timestamp_override or "",
    }

    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        data["node_payload"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["community_centroids"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        if isinstance(data.get("assignment"), dict):
            data["assignment"]["mid_regions"] = {}
    if _estimate_serialised_bytes(data) > MAX_CACHE_BYTES:
        return False

    serialised = json.dumps(data, ensure_ascii=False)

    try:
        key = _cache_encryption_key(store)
        ciphertext = encrypt_field(serialised, key, _CACHE_AAD)
    except (OSError, ValueError, RuntimeError) as exc:
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
    new_gen = advance_generation()
    reset_dirty_counter()
    ts_iso = datetime.now(timezone.utc).isoformat()
    global _rebuild_timestamp_override  # noqa: PLW0603
    with _GEN_LOCK:
        _rebuild_timestamp_override = ts_iso
    result = save(store, assignment, rich_club, node_payload=node_payload, max_degree=max_degree)
    with _GEN_LOCK:
        _rebuild_timestamp_override = ""
    return result


def invalidate(store: Any) -> None:
    path = _cache_path(store)
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.debug("runtime_graph_cache invalidate failed: %s", exc)


def _rebuild_and_save_rgc(store: Any, *, force: bool = False) -> dict:
    from iai_mcp.community import detect_communities
    from iai_mcp.richclub import rich_club_nodes

    with _PERSISTENT_GRAPH_LOCK:
        if not force:
            # Skip the rebuild (and its allocation) only when the cached snapshot
            # is still usable for recall. The read path's own structural source is
            # the authoritative signal: warm iff overlay/normal, cold otherwise.
            # It already folds in no-snapshot / parity / epoch / generation==0 /
            # age+dirty fuse. The dirty counter is a separate write-volume signal,
            # so a cache can be cold while the counter is zero — gate on both.
            try:
                structural_source = load_recall_structural(store)[3]
            except Exception:  # noqa: BLE001 -- a probe failure must never drop a warm-up
                structural_source = "cold_degrade"  # fail toward rebuilding
            cache_is_warm = structural_source in ("overlay", "normal")
            if cache_is_warm and get_dirty_counter() <= _FUSE_DIRTY_THRESHOLD:
                return {
                    "rebuilt": False,
                    "skipped": "warm_and_below_dirty_threshold",
                    "structural_source": structural_source,
                    "node_count": 0,
                    "generation": get_current_generation(),
                }

        graph = _get_persistent_graph()

        try:
            all_records = list(store.all_records())
        except Exception as exc:  # noqa: BLE001
            logger.warning("_rebuild_and_save_rgc: all_records failed: %s", exc)
            raise

        nodes: list = []
        for rec in all_records:
            try:
                nodes.append((
                    rec.id,
                    None,
                    list(rec.embedding),
                    {
                        "embedding": list(rec.embedding),
                        "surface": rec.literal_surface,
                        "centrality": float(rec.centrality),
                        "tier": rec.tier,
                        "pinned": bool(rec.pinned),
                        "tags": list(getattr(rec, "tags", []) or []),
                        "language": str(getattr(rec, "language", "en") or "en"),
                    },
                ))
            except Exception:  # noqa: BLE001
                pass

        edges: list = []
        try:
            edges_df = store.db.open_table("edges").to_pandas()
            if not edges_df.empty:
                from uuid import UUID as _UUID
                for _, row in edges_df.iterrows():
                    try:
                        src = _UUID(str(row["src"]))
                        dst = _UUID(str(row["dst"]))
                        w = float(row.get("weight", 1.0) or 1.0)
                        edges.append((src, dst, w, "hebbian"))
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("_rebuild_and_save_rgc: edge load failed: %s", exc)

        graph.clear_and_rebuild(nodes, edges)

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
