"""Client-facing semantic recall helpers.

Two entry points:

- ``recall_semantic_warm``: client semantic path. The local store is the
  always-available awake memory — semantic recall works daemon-independent.
  This function is only reached AFTER the daemon recall RPC has already
  failed/timed out, so it never depends on the daemon being up.

  When reached, it CONSTRUCTS its OWN Rust embedder synchronously under a
  bypass-safe time budget (a worker thread joined with a timeout covering the
  construct AND one smoke-encode together) and feeds it into the full
  structural-parity path (a local MemoryStore(store_root) + the SAME Layer-1
  structural loader the daemon-up path uses: on-disk mosaic/rich-club cache +
  bounded incident_edges 2-hop spread + uncapped contradicts +
  recall_for_response, all daemon-free over on-disk/SQLite reads), returning
  FULL STRUCTURAL hits tagged ``_source="daemon-down-full"``.

  BYPASS-SAFE FLOOR (non-negotiable): if the construct OR the smoke-encode
  exceeds the budget (a true cold-disk first-ever model load) OR raises, the
  call returns a STORE-backed recency degrade NOW — never empty, never a
  hard-fail, never blocking the full construct. The budget covers construct +
  encode together; on a warm machine the construct is fast (~tens of ms) so
  this path normally delivers semantic hits.

  ANN-only (local-embed + on-disk ANN, no structural enrichment) is the
  last-resort fallback inside the structural-parity path if the local
  MemoryStore or structural cache cannot be opened.

- ``recall_semantic_degraded``: STORE-backed recency/temporal result. Returns
  hits tagged ``_degraded=True``, never empty as a hard-fail, never bank.

Per-operation daemon-dependence (honest framing):
- Recency read: always works; no embedder needed.
- Semantic: works daemon-independent — each awake caller constructs its own
  embedder under the budget guard. On a warm machine this delivers full
  structural hits; on a cold-disk first load it degrades promptly to recency.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Short timeout (ms) for the embed_cue RPC call to the daemon warm accelerator.
# Configurable via IAI_MCP_EMBED_RPC_TIMEOUT_MS.
_DEFAULT_EMBED_RPC_TIMEOUT_MS = 500


def _embed_rpc_timeout_ms() -> int:
    """Return the configured embed_cue RPC timeout in milliseconds."""
    raw = os.environ.get("IAI_MCP_EMBED_RPC_TIMEOUT_MS", "")
    try:
        v = int(raw)
        return max(50, v)
    except (ValueError, TypeError):
        return _DEFAULT_EMBED_RPC_TIMEOUT_MS


# Default time budget (ms) for the in-process embedder construct + one smoke
# encode on the daemon-independent semantic path. A warm-cache construct +
# encode finishes in a few tens of ms; a true cold-disk first-ever model load
# is several seconds. ~1 s separates the two cleanly, so the budget guard lets
# the warm case through and degrades the cold-disk case promptly.
_DEFAULT_CONSTRUCT_BUDGET_MS = 1000


def _construct_budget_ms() -> int:
    """Return the configured construct + smoke-encode budget in milliseconds."""
    raw = os.environ.get("IAI_MCP_EMBED_CONSTRUCT_BUDGET_MS", "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return _DEFAULT_CONSTRUCT_BUDGET_MS


# A short fixed warmup string used for the smoke-encode inside the budget guard.
# English-only, no user content; proves the freshly constructed embedder can
# actually encode before it is handed to the structural pipeline.
_SMOKE_ENCODE_TEXT = "warmup"


def _construct_with_budget(root: "str | Path") -> "tuple[Any, float]":
    """Construct an embedder + run ONE smoke-encode under a join-with-timeout budget.

    The embedder construction releases the GIL inside the native extension and
    is a synchronous, uninterruptible call from the calling thread — so the
    budget CANNOT be an elapsed-check after a blocking construct (that would pay
    the full cold-disk load before degrading). Instead a worker thread does
    BOTH the construct (via the single embedder funnel — the same seam the
    structural path injects through) AND one smoke-encode; the main thread joins
    with a timeout. The embedder is returned ONLY if BOTH the construct and the
    smoke-encode finish within the budget.

    Returns:
        (embedder, worker_ms) on success — construct + smoke-encode both
        completed within the budget.
        (None, elapsed_ms) if the worker is still alive at the timeout (construct
        or encode over budget) OR the construct/encode raised — the caller falls
        to the recency floor. Never raises.
    """
    import time as _time

    box: "dict[str, Any]" = {}

    def _work() -> None:
        t0 = _time.monotonic()
        try:
            import iai_mcp.embed as _embed_mod

            # Resolve the funnel at call time (module attribute) so test stubs
            # that monkeypatch embedder_for_store are honored. Pass None for the
            # store: the funnel treats a dim-less store as the default single
            # English model (the post-warm path opens its own store separately).
            emb = _embed_mod.embedder_for_store(None)
            # ONE smoke-encode inside the SAME bounded worker (the budget covers
            # construct + encode together — not construct alone).
            emb.embed(_SMOKE_ENCODE_TEXT)
            box["emb"] = emb
        except Exception as exc:  # noqa: BLE001 — construct/encode failure: stay floor
            box["err"] = exc
            logger.debug("construct_with_budget_failed: %s", exc)
        finally:
            box["ms"] = (_time.monotonic() - t0) * 1000.0

    th = threading.Thread(target=_work, daemon=True, name="iai-embed-construct")
    t0 = _time.monotonic()
    th.start()
    th.join(timeout=_construct_budget_ms() / 1000.0)
    if th.is_alive() or "emb" not in box:
        # Over budget (construct or encode) OR construct/encode raised → floor.
        return None, (_time.monotonic() - t0) * 1000.0
    return box["emb"], box.get("ms", (_time.monotonic() - t0) * 1000.0)


def _send_embed_cue_rpc(cue: str, timeout_ms: int) -> "list[float] | None":
    """Send an embed_cue control message to the daemon warm accelerator.

    Returns the 384-d embedding vector on success, None on timeout/unavailable.
    Uses the same socket path as _send_socket_request in cli.py.
    Does NOT acquire any flock; the daemon embed_cue handler holds none either.
    """
    import asyncio
    import json

    from iai_mcp.concurrency import SOCKET_PATH

    sock_path = os.environ.get("IAI_DAEMON_SOCKET_PATH") or str(SOCKET_PATH)
    connect_timeout = timeout_ms / 1000.0

    async def _runner() -> "list[float] | None":
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(sock_path),
                timeout=connect_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError, asyncio.TimeoutError):
            return None
        try:
            req = {"type": "embed_cue", "cue": cue}
            writer.write((json.dumps(req) + "\n").encode("utf-8"))
            await writer.drain()
            line = await asyncio.wait_for(
                reader.readline(),
                timeout=connect_timeout,
            )
            if not line:
                return None
            resp = json.loads(line.decode("utf-8"))
            if not isinstance(resp, dict) or not resp.get("ok"):
                return None
            vec = resp.get("embedding")
            if not isinstance(vec, list) or len(vec) != 384:
                return None
            return [float(x) for x in vec]
        except Exception:  # noqa: BLE001
            return None
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except OSError:
                pass

    try:
        return asyncio.run(_runner())
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Daemon-independent semantic recall.
#
# The local store is the always-available awake memory: semantic recall does
# NOT depend on the daemon. When reached (the daemon recall RPC has already
# failed), recall constructs its own embedder under the join-with-timeout budget
# guard (_construct_with_budget above) and runs the SAME Layer-1 structural
# parity path the daemon-up path uses:
# - a local MemoryStore(store_root) (reused via the cached handle below),
# - the on-disk mosaic/rich-club structural loader + bounded incident_edges
# 2-hop spread + uncapped contradicts + recall_for_response,
# - all daemon-free (on-disk / SQLite reads).
# Returns FULL STRUCTURAL hits. On a true cold-disk construct (over budget) OR
# any construct/encode failure, recall returns a STORE-backed recency degrade
# NOW — the bypass-safe floor, never empty, never a hard-fail.
#
# The local MemoryStore handle is cached at module level so the open cost
# (SQLite connect + ANN index load) is paid at most once per process for the
# daemon-independent structural path.
# ---------------------------------------------------------------------------

# Cached local MemoryStore handle for the daemon-independent structural path.
# None until successfully opened.
_WARM_LOCAL_STORE: "Any" = None


def _get_or_open_warm_local_store(store_root: Path) -> "Any":
    """Return the cached local MemoryStore, constructing it once if needed.

    Caches the store handle at module level so the HippoDB open cost (SQLite
    connect + ANN index load) is paid at most once per process for the
    daemon-independent structural path.

    Returns None on any failure (caller falls back to ANN-only or degrade).
    """
    global _WARM_LOCAL_STORE  # noqa: PLW0603
    if _WARM_LOCAL_STORE is not None:
        return _WARM_LOCAL_STORE
    try:
        from iai_mcp.store import MemoryStore
        store = MemoryStore(str(store_root))
        _WARM_LOCAL_STORE = store
        return store
    except Exception as exc:  # noqa: BLE001 — store open failure: fall back to ANN-only
        logger.debug("local_store_open_failed: %s", exc)
        return None


def _recall_daemon_down_post_warm(
    store_root: Path,
    cue: str,
    embedder: "Any",
    n: int,
    session_id: "str | None",
    profile_state: "dict | None" = None,
) -> "list[dict]":
    """Post-warm daemon-down semantic recall with FULL STRUCTURAL PARITY.

    Routes through core.dispatch with the warm local embedder injected into
    iai_mcp.embed.embedder_for_store — this is the SAME Layer-1 structural
    path the daemon-up recall uses (ANN + bounded incident_edges 2-hop +
    uncapped contradicts + load_recall_structural + recall_for_response),
    all daemon-free (on-disk mosaic/rich-club cache + SQLite reads).

    The local store IS the always-available awake memory; daemon-down gets
    the same structural enrichment as daemon-up. Using core.dispatch ensures
    EXACT parity with the daemon-up pipeline (same ranked scoring, same
    profile_state, same arousal routing) — not a hand-rolled copy that
    could diverge in scoring config.

    Embedder injection is thread-safe via a module-level override in
    iai_mcp.embed that is set immediately before dispatch and cleared
    in the finally block. This is the same override slot used by the
    daemon's own embedder injection (embedder_for_store reads it).

    Falls back to LOCAL ANN-only if the local MemoryStore cannot be
    opened. Falls back to recency degrade as last resort.
    """
    from iai_mcp.hippo import (
        degraded_semantic_recall as _degrade,
        EMBED_DIM,
    )

    # Verify the warm embedder can embed (guards against corrupted instances).
    try:
        _test_vec = embedder.embed(cue)
        if not isinstance(_test_vec, (list, tuple)) or len(_test_vec) != EMBED_DIM:
            raise ValueError(f"embed returned unexpected dim {len(_test_vec) if hasattr(_test_vec, '__len__') else '?'}")
    except Exception as exc:  # noqa: BLE001
        logger.debug("daemon_down_local_embed_failed: %s", exc)
        rows = _degrade(store_root, cue, limit=n, session_id=session_id)
        return _stamp_degrade_source(rows)

    # Try to open (or reuse) the local MemoryStore.
    local_store = _get_or_open_warm_local_store(store_root)
    if local_store is None:
        # Local store unavailable: ANN-only fallback.
        return _ann_only_daemon_down(store_root, list(_test_vec), n, cue, session_id)

    # Full structural parity via core.dispatch with the warm embedder injected.
    # The dispatch routes the full Layer-1 pipeline (same path as daemon-up)
    # using the local MemoryStore and the warm local embedder.
    try:
        import iai_mcp.embed as _embed_mod
        from iai_mcp import core as _core_mod

        # Inject the warm embedder: temporarily replace embedder_for_store
        # to return the warm singleton for this call. Restored in the
        # finally block regardless of outcome.
        # Note: this mutates a module-level attribute and restores it in
        # finally. This is safe on the iai recall CLI path (single caller
        # per process invocation) but NOT safe if multiple in-process callers
        # call _recall_daemon_down_post_warm concurrently. The CLI path is
        # always single-caller; the daemon-up path never reaches this function.
        # This is the cleanest injection point because embedder_for_store
        # is the single factory consulted by core.dispatch for all recalls.
        _captured_embedder = embedder  # close over the warm singleton

        def _injected_embedder_for_store(_store: "Any") -> "Any":
            return _captured_embedder

        _orig_efs = _embed_mod.embedder_for_store
        try:
            _embed_mod.embedder_for_store = _injected_embedder_for_store
            params = {
                "cue": cue,
                "session_id": session_id or "daemon-down",
                "budget_tokens": n * 300,
            }
            resp_dict = _core_mod.dispatch(local_store, "memory_recall", params)
        finally:
            _embed_mod.embedder_for_store = _orig_efs

        hits_raw = resp_dict.get("hits") or []
        if hits_raw:
            return [
                {
                    "literal_surface": h.get("literal_surface", "") or h.get("surface", "") or "",
                    "score": float(h.get("score") or h.get("final_score") or 0.0),
                    "_source": "daemon-down-full",
                    "record_id": str(h.get("record_id") or h.get("id") or ""),
                }
                for h in hits_raw[:n]
            ]
        # dispatch returned empty — fall through to ANN-only.
    except Exception as exc:  # noqa: BLE001 — structural path failure: ANN-only fallback
        logger.debug("daemon_down_structural_path_failed: %s", exc)

    # LAST-RESORT: local ANN-only (no structural enrichment).
    return _ann_only_daemon_down(store_root, list(_test_vec), n, cue, session_id)


def _ann_only_daemon_down(
    store_root: Path,
    vec: list[float],
    n: int,
    cue: str,
    session_id: "str | None",
) -> "list[dict]":
    """LAST-RESORT cold-start daemon-down path: local ANN + SQLite fetch, no structural.

    Used when the local MemoryStore or structural cache cannot be opened.
    ANN-only: returns records by cosine proximity only, no 2-hop spread or
    rich-club bias. Hub-sensitive gold reachable only via structural spread
    may be MISSED on this path — it is the cold-start / last-resort fallback,
    NOT normal post-warm behaviour.
    """
    from iai_mcp.hippo import (
        degraded_semantic_recall as _degrade,
        EMBED_DIM,
    )
    from iai_mcp.hippo import _ann_lookup_client

    try:
        labels = _ann_lookup_client(store_root, vec, k=n, embed_dim=EMBED_DIM)
        if labels:
            rows = _fetch_records_by_labels(store_root, labels, n=n)
            if rows:
                return rows
    except Exception:  # noqa: BLE001
        pass

    # ANN also failed: recency degrade.
    rows = _degrade(store_root, cue, limit=n, session_id=session_id)
    return _stamp_degrade_source(rows)


def _stamp_degrade_source(rows: "list[dict]") -> "list[dict]":
    """Relabel recency-degrade rows from 'direct-store' to 'daemon-down-degrade'.

    MEDIUM-3: the cold-start degrade window is distinguishable from the
    post-warm full-semantic result and from other store-backed paths.
    This relabelling happens at the recall_semantic_warm boundary ONLY;
    degraded_semantic_recall's own 'direct-store' contract is UNCHANGED so
    other callers of degraded_semantic_recall are not affected.
    """
    result = []
    for row in rows:
        stamped = dict(row)
        stamped["_source"] = "daemon-down-degrade"
        result.append(stamped)
    return result


def recall_semantic_warm(
    store_root: "str | Path",
    cue: str,
    n: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    """Daemon-independent semantic recall using a self-constructed embedder.

    This function runs in the CLIENT process and is only reached AFTER the
    daemon recall RPC has already failed/timed out — so it never depends on the
    daemon being up. The local store is the always-available awake memory:
    semantic recall works daemon-independent.

    1. Construct an embedder + run one smoke-encode synchronously, under a
       join-with-timeout budget guard (_construct_with_budget): a worker thread
       does the construct (via the single embedder funnel) AND one smoke-encode;
       the main thread joins with a timeout. The embedder is used ONLY if both
       finish within the budget.

    2. If the embedder is ready: feed it into the full structural-parity path
       (_recall_daemon_down_post_warm) — a local MemoryStore(store_root) + the
       SAME Layer-1 structural loader the daemon-up path uses (on-disk
       mosaic/rich-club cache + bounded incident_edges 2-hop spread + uncapped
       contradicts + recall_for_response). Returns FULL STRUCTURAL hits tagged
       ``daemon-down-full``, daemon-free. ANN-only is the last-resort fallback
       inside that path if the structural cache cannot be opened.

    3. BYPASS-SAFE FLOOR: if the construct OR the smoke-encode is over budget (a
       true cold-disk first-ever model load) OR raises, return a STORE-backed
       recency degrade NOW — never empty, never a hard-fail, never blocking the
       full construct.

    The daemon memory_recall handler is NOT called here.
    """
    from iai_mcp.hippo import degraded_semantic_recall as _degrade

    root = Path(store_root)

    # Construct an embedder + smoke-encode under the bypass-safe budget guard.
    embedder, _construct_ms = _construct_with_budget(root)
    if embedder is None:
        # Cold-disk over budget OR construct/encode raised → recency floor NOW.
        rows = _degrade(root, cue, limit=n, session_id=session_id)
        # Observability (best-effort): the construct fell to the recency floor.
        # Emit a cue-DERIVED metric only — a scrubbed fixed reason token, the
        # construct time, and the source. NEVER the cue or any cue-derived
        # substring (no exception string is forwarded here). A telemetry
        # failure must not change this degrade-path return value or timing, so
        # we read the cached store handle directly (NEVER open one just to emit)
        # and fall back to a debug log when no store is open.
        _emit_recall_source(
            "recency-degrade",
            construct_ms=_construct_ms,
            reason="construct_timeout_or_fail",
            session_id=session_id,
        )
        return _stamp_degrade_source(rows)

    # Embedder ready: full structural parity, daemon-free.
    result = _recall_daemon_down_post_warm(
        root, cue, embedder, n, session_id, profile_state=None
    )
    # Observability (best-effort): distinguish a true structural / ANN-only hit
    # (the embedder WAS used) from an internal fall-through to the recency floor
    # (which _stamp_degrade_source tags "daemon-down-degrade"). Only the
    # recency fall-through counts toward fallback_rate; ANN-only and structural
    # both used the constructed embedder → "semantic-inprocess".
    _fell_to_recency = (not result) or all(
        r.get("_source") == "daemon-down-degrade" for r in result
    )
    if _fell_to_recency:
        _emit_recall_source(
            "recency-degrade",
            construct_ms=_construct_ms,
            reason="construct_timeout_or_fail",
            session_id=session_id,
        )
    else:
        # construct_ms covers construct + smoke-encode (the budget worker). The
        # dispatch encode lives inside _recall_daemon_down_post_warm (left
        # UNEDITED), so encode_ms is captured at the daemon-UP site instead.
        _emit_recall_source(
            "semantic-inprocess",
            construct_ms=_construct_ms,
            session_id=session_id,
        )
    return result


def _emit_recall_source(
    source: str,
    *,
    construct_ms: "float | None" = None,
    encode_ms: "float | None" = None,
    reason: "str | None" = None,
    session_id: "str | None" = None,
) -> None:
    """Best-effort recall-source telemetry for the daemon-down construct path.

    Observability ONLY — wrapped end-to-end so it can NEVER raise out and never
    change the recall return value or timing materially (the bypass-safe floor).

    The deepest CLI degrade may have NO open store. We read the cached local
    store handle DIRECTLY (the module global set by the post-warm path) and
    NEVER open one just to emit — opening a store here would add a
    SQLite-connect + ANN-load side effect to the degrade path. When no store
    is reachable, emit_best_effort logs a stdlib debug line with the same
    (cue-DERIVED, scrubbed) metrics instead.

    Payload carries cue-DERIVED metrics only: source / construct_ms / encode_ms
    and a SCRUBBED fixed reason token — NEVER the raw cue or a cue-derived
    substring, and no exception string.
    """
    try:
        from iai_mcp.events import emit_best_effort, TELEMETRY_RECALL_SOURCE

        data: "dict[str, Any]" = {"source": source}
        if construct_ms is not None:
            data["construct_ms"] = round(float(construct_ms), 2)
        if encode_ms is not None:
            data["encode_ms"] = round(float(encode_ms), 2)
        if reason is not None:
            data["reason"] = reason
        emit_best_effort(
            _WARM_LOCAL_STORE,
            TELEMETRY_RECALL_SOURCE,
            data,
            severity="info",
            session_id=session_id or "-",
        )
    except Exception:  # noqa: BLE001 -- telemetry must never break recall
        try:
            logger.debug("recall_source_emit_failed source=%s", source)
        except Exception:  # noqa: BLE001
            pass


def _fetch_records_by_labels(
    store_root: "str | Path",
    vec_labels: "list[int]",
    n: int = 10,
) -> "list[dict]":
    """Fetch records by hnswlib vec_label from SQLite and decrypt surfaces.

    CLIENT primitive — opens HippoDB SHARED, reads identified rows.
    Returns [] on any error.
    """
    from iai_mcp.hippo import AccessMode, HippoDB

    if not vec_labels:
        return []

    root = Path(store_root)
    db: "HippoDB | None" = None
    try:
        db = HippoDB(root, access_mode=AccessMode.SHARED, read_only=True, _lock_timeout_override=0.25)

        # Decrypt surfaces using the same key derivation as MemoryStore.
        _crypto_key: "bytes | None" = None
        try:
            from iai_mcp.crypto import CryptoKey as _CK
            _crypto_key = _CK(store_root=root).get_or_create()
        except Exception:  # noqa: BLE001
            pass

        try:
            from iai_mcp.crypto import decrypt_field as _df, is_encrypted as _ie
        except Exception:  # noqa: BLE001
            _df = None  # type: ignore[assignment]
            _ie = None  # type: ignore[assignment]

        results: list[dict] = []
        for label in vec_labels[:n]:
            with db._conn_lock:
                row = db._conn.execute(
                    "SELECT id, literal_surface, created_at FROM records"
                    " WHERE vec_label = ? AND tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0",
                    (label,),
                ).fetchone()
            if row is None:
                continue
            row_id = str(row["id"] or "")
            surface = row["literal_surface"] or ""
            if surface and _crypto_key is not None and _ie is not None and _df is not None:
                try:
                    if _ie(surface):
                        surface = _df(surface, _crypto_key, row_id.encode("utf-8"))
                except Exception:  # noqa: BLE001
                    pass
            results.append({
                "literal_surface": surface,
                "score": 1.0,
                "_source": "direct-store",
            })
        return results
    except Exception:  # noqa: BLE001
        return []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass


def recall_semantic_degraded(
    store_root: "str | Path",
    cue: str,
    n: int = 10,
    *,
    session_id: "str | None" = None,
) -> "list[dict]":
    """Degraded store-backed recall when no warm embedder is available.

    Opens HippoDB(SHARED, read_only=True) in the CLIENT process and returns a
    functional recency/temporal result tagged ``_degraded=True``.

    Never empty as a hard-fail, never bank. This is the STORE-backed degraded path.
    """
    from iai_mcp.hippo import degraded_semantic_recall as _degrade
    return _degrade(store_root, cue, limit=n, session_id=session_id)
