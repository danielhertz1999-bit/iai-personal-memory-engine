"""LanceDB-backed persistent memory store ( storage engine, sync write).

tables:
- `records`: MemoryRecord rows (one per memory).
- `edges`: (src, dst, edge_type, weight, updated_at) -- Hebbian + contradicts edges.

additions :
- `events`: all runtime state (S4 contradictions, trajectory metrics, alerts,
  llm_health, schema_induction_run, cls_consolidation_run, etc.).
- `budget_ledger`: D-GUARD per-day USD spend by kind (BudgetLedger).
- `ratelimit_ledger`: D-GUARD 429 history for 15-min cooldown (RateLimitLedger).

: NO scattered .jsonl or .json files. Every runtime event lives here.

Embedding dimension: (English-Only Brain pivot) defaults to
`bge-small-en-v1.5` (384d). The records schema reads the configured dimension
from `iai_mcp.embed.DEFAULT_DIM` at first table creation. Stores created during
the brief Phase-2 era still carry 1024d embeddings and stay readable via
`embedder_for_store(store)` until the user re-embeds them down to 384d.

Storage root defaults to `~/.iai-mcp/lancedb` ( local-first), overridable
via IAI_MCP_STORE env var or the `path` constructor argument.

encryption-at-rest :
- literal_surface / provenance_json / profile_modulation_gain_json on records
  table are AES-256-GCM encrypted with a key sourced from the OS keychain.
- events.data_json on events table is also encrypted.
- Embeddings / tags / language / schema_version / timestamps stay plaintext.
- Encryption is transparent to callers: store.insert() encrypts and
  store.get() decrypts; no change to the MemoryRecord dataclass.
- AD = record UUID bytes, binding ciphertext to its row so cut-and-paste
  attacks fail on decrypt.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import json
import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Callable
from uuid import UUID

import lancedb
import pyarrow as pa

# W5: cached AESGCM cipher per store; reuse safe per
# https://cryptography.io/en/latest/hazmat/primitives/aead/ — single AESGCM
# can be reused across operations as long as nonces are unique. We use random
# per-record nonces in encrypt_field, so cache reuse is correct.
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from iai_mcp.crypto import (
    CIPHERTEXT_PREFIX,
    NONCE_BYTES,
    CryptoKey,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import (
    DEFAULT_EMBED_DIM,
    EMBED_DIM,
    SCHEMA_VERSION_CURRENT,
    MemoryRecord,
    TIER_ENUM,
)

DEFAULT_STORAGE_PATH = Path.home() / ".iai-mcp"

# tables
RECORDS_TABLE = "records"
EDGES_TABLE = "edges"

# tables
EVENTS_TABLE = "events"
BUDGET_TABLE = "budget_ledger"
RATELIMIT_TABLE = "ratelimit_ledger"

# edge type enum. = {hebbian, contradicts}. adds 6.
# consolidated_from   -- CLS sleep cycle semantic <- source episodes
# schema_instance_of  -- schema induction episode -> schema hub
# temporal_next       -- record insert (same session) episode chain
# invariant_anchor    -- S5 kernel stable-fact permanent hub
# curiosity_bridge    -- LEARN-04 question -> triggering records
# profile_modulates   -- profile knob runtime gain
# hebbian_structure -- TEM factorization LTP on structure edges
EDGE_TYPES: frozenset[str] = frozenset({
    "hebbian",
    "contradicts",
    "consolidated_from",
    "schema_instance_of",
    "temporal_next",
    "invariant_anchor",
    "curiosity_bridge",
    "profile_modulates",
    "hebbian_structure",
})

# RFC-4122 canonical UUID regex. Accept both str and UUID inputs; reject anything
# that could embed a SQL-like escape. Hoisted to module scope so the pattern
# object is compiled once.
_UUID_STR_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _uuid_literal(value: UUID | str) -> str:
    """Return a LanceDB WHERE-safe UUID literal.

    H-01: callers previously interpolated UUIDs into `where=` predicates via bare
    f-strings. Safe today (inputs are UUID objects), but the pattern propagates
    risk as tag-based filtering arrives in This helper normalises any
    UUID (object or canonical str) into its canonical lowercase form and rejects
    anything that does not match the RFC-4122 shape, so the f-string cannot
    carry injection content.
    """
    s = str(value).lower()
    if not _UUID_STR_RE.match(s):
        raise ValueError(f"not a canonical UUID: {value!r}")
    return s


# local dim table so store creation does NOT pull in
# iai_mcp.embed (and by extension sentence_transformers + torch). The
# keys stay in sync with iai_mcp.embed.MODEL_REGISTRY by convention;
# if a new model is added there, add it here too. Duplicating the
# table keeps the torch import on the Embedder() construction path
# only, saving ~500 MB of RSS for code paths that just want to read
# from the store.
_STORE_LOCAL_DIM_TABLE: dict[str, int] = {
    "bge-m3": 1024,
    "multilingual-e5-small": 384,
    "bge-small-en-v1.5": 384,
}


def _resolve_embed_dim() -> int:
    """Pick the embedding dimension for the records table on first creation.

    Priority:
    1. Environment override IAI_MCP_EMBED_DIM (test hermeticity / migration dry-runs)
    2. IAI_MCP_EMBED_MODEL env var -> dim via local table
    3. types.DEFAULT_EMBED_DIM (reflects the module-level default;
       flipped it from 1024 to 384 to match the PROJECT.md
       spec of bge-small-en-v1.5)

    this function does NOT import iai_mcp.embed any more.
    That import eagerly loads sentence_transformers + torch, adding
    ~500 MB of RSS to every MemoryStore() creation even when the code
    path never embeds anything. The local dim table mirrors
    embed.MODEL_REGISTRY; we only pay the torch cost when an Embedder
    is actually instantiated.
    """
    env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
    if env_dim:
        try:
            return int(env_dim)
        except ValueError:
            pass
    env_key = os.environ.get("IAI_MCP_EMBED_MODEL")
    if env_key and env_key in _STORE_LOCAL_DIM_TABLE:
        return _STORE_LOCAL_DIM_TABLE[env_key]
    return DEFAULT_EMBED_DIM


class MemoryStore:
    """Embedded LanceDB wrapper.

    design: sync writes, single-user, local filesystem.
    adds events/budget_ledger/ratelimit_ledger tables
    plus v2 MemoryRecord fields (language, s5_trust_score, profile_modulation_gain,
    schema_version). Existing rows remain readable with schema_version=1.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        user_id: str = "default",
        read_consistency_interval: timedelta | None = None,
    ) -> None:
        """Open (or initialise) a LanceDB-backed store.

        ``read_consistency_interval`` controls how often the shared
        connection re-checks for commits made by other processes
        against the same ``lancedb`` directory. See
        https://docs.lancedb.com/tables/consistency.

        - ``None`` (default) — no auto-refresh. Correct for short-lived
          MCP tool calls that open the store, do one write/read, and
          exit; they always see the latest manifest because each call
          is a fresh connection.
        - ``timedelta(seconds=0)`` — strong consistency. Every read
          re-checks the latest version. Correct for the sleep daemon
          (long-lived process running alongside MCP writers).
        - ``timedelta(seconds=N)`` — eventual consistency with ``N``
          seconds of staleness tolerance. Use when read traffic is
          heavy and an N-second lag is acceptable.
        """
        env_path = os.environ.get("IAI_MCP_STORE")
        self.root = Path(env_path) if env_path else Path(path or DEFAULT_STORAGE_PATH)
        self.root.mkdir(parents=True, exist_ok=True)
        self._read_consistency_interval: timedelta | None = read_consistency_interval
        connect_kwargs: dict[str, object] = {}
        if read_consistency_interval is not None:
            connect_kwargs["read_consistency_interval"] = read_consistency_interval
        self.db = lancedb.connect(str(self.root / "lancedb"), **connect_kwargs)
        # Resolve the embedding dimension once so records table + insert guard agree.
        self._embed_dim: int = _resolve_embed_dim()
        self._ensure_tables()
        # encryption-at-rest. Per-store user_id for multi-tenant layout.
        # The key is loaded lazily on first encrypt/decrypt so test suites that
        # create hundreds of temporary MemoryStore() instances don't each hit the
        # (mocked or real) keyring backend on __init__.
        self._user_id: str = user_id
        self._crypto_key_wrapper: CryptoKey = CryptoKey(user_id=user_id, store_root=self.root)
        self._crypto_key: bytes | None = None
        # optional store -> runtime-graph sync callback. Set by
        # retrieve.build_runtime_graph via register_graph_sync_hook(). Every
        # insert / update / delete fires this hook inside try/except so the
        # LanceDB write remains authoritative — a buggy or absent hook can
        # never break the store.
        self._graph_sync_hook: Callable[[str, "MemoryRecord"], None] | None = None
        # optional async write queue. When live, insert() routes
        # through it; when None, insert() uses the legacy sync path. The
        # event loop runs on a dedicated background thread so sync callers
        # can dispatch via asyncio.run_coroutine_threadsafe.
        self._write_queue = None  # type: ignore[assignment]
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self._async_conn = None  # lancedb AsyncConnection
        # optional async provenance queue. When set, writes
        # routed through queue_provenance_batch go off the recall
        # critical path; when None we fall back to the sync
        # append_provenance_batch call (back-compat).
        self._provenance_queue = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ schema

    def _ensure_tables(self) -> None:
        # records table schema uses the configured embedder dimension.
        # For a pre-existing table created under a different dim we honour the
        # existing schema; migration will rewrite it when the user opts in.
        records_schema = pa.schema(
            [
                ("id", pa.string()),
                ("tier", pa.string()),
                ("literal_surface", pa.string()),
                ("aaak_index", pa.string()),
                ("embedding", pa.list_(pa.float32(), self._embed_dim)),
                # TEM factorization (Whittington-Behrens 2020).
                # D=10000 BSC packed bits = 1250 bytes; plaintext like `embedding`
                # because it is part of the retrieval surface.
                # Renamed v3 -> v4 from the legacy `hd_vector_json` (pa.string())
                # column via migrate_hd_vector_to_structure_hv_v3_to_v4.
                ("structure_hv", pa.binary()),
                ("community_id", pa.string()),
                ("centrality", pa.float32()),
                ("detail_level", pa.int32()),
                ("pinned", pa.bool_()),
                ("stability", pa.float32()),
                ("difficulty", pa.float32()),
                ("last_reviewed", pa.timestamp("us", tz="UTC")),
                ("never_decay", pa.bool_()),
                ("never_merge", pa.bool_()),
                ("provenance_json", pa.string()),
                ("created_at", pa.timestamp("us", tz="UTC")),
                ("updated_at", pa.timestamp("us", tz="UTC")),
                ("tags_json", pa.string()),
                # v2 columns (D-02a / prep / / )
                ("language", pa.string()),                    # ISO-639-1 tag
                ("s5_trust_score", pa.float32()),             # prep, default 0.5
                ("profile_modulation_gain_json", pa.string()),# runtime gain map
                ("schema_version", pa.int32()),               # migration marker
            ]
        )
        if RECORDS_TABLE not in self._table_names():
            self.db.create_table(RECORDS_TABLE, schema=records_schema)
        else:
            # Existing table: inspect its schema and adjust _embed_dim to match
            # so legacy stores (384d bge-small) keep working until migration.
            try:
                tbl = self.db.open_table(RECORDS_TABLE)
                arrow_schema = tbl.schema
                emb_field = arrow_schema.field("embedding")
                # pa.list_(..., N) fixed-size list -> .type.list_size
                actual_dim = getattr(emb_field.type, "list_size", None)
                if actual_dim and int(actual_dim) > 0:
                    self._embed_dim = int(actual_dim)
            except Exception:
                pass

        edges_schema = pa.schema(
            [
                ("src", pa.string()),
                ("dst", pa.string()),
                ("edge_type", pa.string()),
                ("weight", pa.float32()),
                ("updated_at", pa.timestamp("us", tz="UTC")),
            ]
        )
        if EDGES_TABLE not in self._table_names():
            self.db.create_table(EDGES_TABLE, schema=edges_schema)

        # --------- events table (single source of runtime state)
        events_schema = pa.schema(
            [
                ("id", pa.string()),                             # UUID str
                ("kind", pa.string()),                           # s4_contradiction | ...
                ("severity", pa.string()),                       # info | warning | critical | ""
                ("domain", pa.string()),                         # monotropic domain | ""
                ("ts", pa.timestamp("us", tz="UTC")),
                ("data_json", pa.string()),                      # kind-specific payload (JSON)
                ("session_id", pa.string()),
                ("source_ids_json", pa.string()),                # JSON array of UUID strs
            ]
        )
        if EVENTS_TABLE not in self._table_names():
            self.db.create_table(EVENTS_TABLE, schema=events_schema)

        # --------- D-GUARD BudgetLedger table
        budget_schema = pa.schema(
            [
                ("date", pa.string()),                           # YYYY-MM-DD UTC
                ("usd_spent", pa.float32()),
                ("kind", pa.string()),                           # "llm" | "batch" | ...
                ("ts", pa.timestamp("us", tz="UTC")),
            ]
        )
        if BUDGET_TABLE not in self._table_names():
            self.db.create_table(BUDGET_TABLE, schema=budget_schema)

        # --------- D-GUARD RateLimitLedger table
        ratelimit_schema = pa.schema(
            [
                ("ts", pa.timestamp("us", tz="UTC")),
                ("status_code", pa.int32()),                     # typically 429
                ("endpoint", pa.string()),                       # "anthropic" | ...
            ]
        )
        if RATELIMIT_TABLE not in self._table_names():
            self.db.create_table(RATELIMIT_TABLE, schema=ratelimit_schema)

    def _table_names(self) -> list[str]:
        """Version-agnostic shim: lancedb 0.30+ returns a paginated response from
        `list_tables()` whose `.tables` attr is the list; older versions returned the
        list directly via the deprecated `table_names()` method.
        """
        result = self.db.list_tables()
        if hasattr(result, "tables"):
            return list(result.tables)
        return list(result)

    @property
    def embed_dim(self) -> int:
        """Actual embedding dimension in the records table."""
        return self._embed_dim

    @property
    def user_id(self) -> str:
        """user_id that scopes the encryption key (multi-tenant ready)."""
        return self._user_id

    # -------------------------------------------------------------- encryption

    def _key(self) -> bytes:
        """Lazy-load the encryption key from keyring."""
        if self._crypto_key is None:
            self._crypto_key = self._crypto_key_wrapper.get_or_create()
        return self._crypto_key

    def _ad(self, record_id: UUID | str) -> bytes:
        """Associated data for a record's encrypted fields: canonical UUID str bytes.

        Binds ciphertext to its row. An attacker who swaps ciphertext between
        rows (copy row A's literal_surface into row B on disk) will fail to
        decrypt because AD(B) != AD(A) -- InvalidTag.
        """
        return _uuid_literal(record_id).encode("ascii")

    def _encrypt_for_record(self, record_id: UUID, value: str) -> str:
        """Encrypt a per-record sensitive field; idempotent on already-encrypted input."""
        if is_encrypted(value):
            return value
        return encrypt_field(value, self._key(), associated_data=self._ad(record_id))

    @functools.cached_property
    def _cached_aesgcm(self) -> AESGCM:
        """W5: one AESGCM cipher per store lifetime.

        Materialised lazily on first access. Reused across all
        :meth:`_decrypt_for_record` calls — saves the per-call ``AESGCM(key)``
        construction cost (16210 calls per ``_tier0_schema_surfacing`` invocation
        on the 8105-record store before W5).

        Cache invalidation: if ``self._key()`` rotates (key rotation event),
        callers must invoke :meth:`_invalidate_aesgcm_cache`.
        accepts "no rotation during phase" — rotation hook is future work.
        """
        return AESGCM(self._key())

    def _invalidate_aesgcm_cache(self) -> None:
        """Drop the cached AESGCM. Next access re-materialises against current key.

        Reserved for future key-rotation events . Not invoked
        by itself.
        """
        self.__dict__.pop("_cached_aesgcm", None)

    def _decrypt_for_record(self, record_id: UUID, value: str) -> str:
        """Decrypt a per-record sensitive field; pass through plaintext unchanged.

        Back-compat: pre-02-08 rows are plaintext -- return them as-is so
        readers see the same data until v2->v3 migration re-encrypts them.

        W5: uses :attr:`_cached_aesgcm` instead of
        constructing a fresh ``AESGCM(key)`` on every call. Per-call cost
        drops from ``AESGCM.__init__ + AESGCM.decrypt`` to ``AESGCM.decrypt``
        only. ``crypto.decrypt_field`` is intentionally NOT modified ( —
        keep crypto.py decoupled + stateless for callers that pass key bytes
        directly).
        """
        if not is_encrypted(value):
            return value
        if not value.startswith(CIPHERTEXT_PREFIX):
            # Defensive: is_encrypted() should already have guaranteed this.
            raise ValueError("field is not iai:enc:v1:-prefixed ciphertext")
        payload_b64 = value[len(CIPHERTEXT_PREFIX):]
        payload = base64.b64decode(payload_b64)
        if len(payload) < NONCE_BYTES + 16:  # nonce + minimum GCM tag
            raise ValueError("ciphertext payload too short")
        nonce = payload[:NONCE_BYTES]
        ct_with_tag = payload[NONCE_BYTES:]
        associated_data = self._ad(record_id)
        plaintext_bytes = self._cached_aesgcm.decrypt(
            nonce, ct_with_tag, associated_data or None
        )
        return plaintext_bytes.decode("utf-8")

    # -------------------------------------------------------------------- I/O

    # ------------------------------------------------------- hook

    def register_graph_sync_hook(
        self, hook: Callable[[str, MemoryRecord], None] | None
    ) -> None:
        """register a callback that mirrors store writes to
        the runtime NetworkX graph.

        The hook is called with ``(op, record)`` after every successful
        LanceDB write where ``op`` is one of ``"insert" | "update" |
        "delete"``. Hook exceptions are caught and logged to stderr as
        a structured JSON ``graph_sync_failed`` event; the store write
        is authoritative and never rolled back on hook failure.

        Idempotent — passing a new callable replaces the previous hook;
        passing ``None`` unregisters it.
        """
        self._graph_sync_hook = hook

    def _fire_graph_sync_hook(self, op: str, record: MemoryRecord) -> None:
        """Dispatch the (op, record) event. Failures are swallowed +
        logged. Never raises."""
        hook = self._graph_sync_hook
        if hook is None:
            return
        try:
            hook(op, record)
        except Exception as exc:
            try:
                sys.stderr.write(
                    json.dumps({
                        "event": "graph_sync_failed",
                        "op": op,
                        "record_id": str(getattr(record, "id", "")),
                        "error": str(exc),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                    + "\n"
                )
            except Exception:
                # Structured logging itself failing must not propagate.
                pass

    def insert(self, record: MemoryRecord) -> None:
        """Append a record. verbatim, no rewrite at write time.

        sensitive fields are encrypted in _to_row before the
        row hits LanceDB. Decryption happens in get()/_from_row for callers.

        : if record.structure_hv is empty bytes (the
        pre-migration sentinel), compute it via tem.bind_structure(record)
        before persisting. This is the autopoietic write-time fill -- the
        record carries its own structural fingerprint into LanceDB so the
        memory_recall_structural branch can rank it without re-derivation.

        fires the optional ``_graph_sync_hook`` after the
        LanceDB write lands so the runtime graph stays in sync with the
        store. Hook failures are logged, never raised.

        if ``enable_async_writes()`` has been called the
        insert is routed through the coalescing AsyncWriteQueue and this
        call blocks until the batch containing ``record`` has flushed to
        disk. Graph-sync fires from the queue's ``on_flushed`` callback,
        so this path still preserves 05-12 semantics.
        """
        if record.tier not in TIER_ENUM:
            raise ValueError(f"invalid tier {record.tier!r}")
        if len(record.embedding) != self._embed_dim:
            raise ValueError(
                f"embedding must be {self._embed_dim}d, got {len(record.embedding)}"
            )
        # lazy structure_hv fill via tem.bind_structure.
        if not record.structure_hv:
            from iai_mcp.tem import bind_structure
            record.structure_hv = bind_structure(record)

        # async-mode route. The queue's coalesce window batches
        # concurrent inserts; run_coroutine_threadsafe + fut.result() give
        # us the same "returns after disk flush" contract as the sync path.
        if self._write_queue is not None and self._async_loop is not None:
            coro = self._write_queue.enqueue(record)
            submit = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
            fut = submit.result()
            # fut is an asyncio.Future owned by the background loop; we
            # need to wait on it from this (sync) thread too.
            done_event = threading.Event()
            result_box: dict = {}

            def _watch(_f: asyncio.Future) -> None:
                if _f.cancelled():
                    result_box["exc"] = asyncio.CancelledError()
                elif _f.exception() is not None:
                    result_box["exc"] = _f.exception()
                else:
                    result_box["val"] = _f.result()
                done_event.set()

            self._async_loop.call_soon_threadsafe(fut.add_done_callback, _watch)
            done_event.wait()
            if "exc" in result_box:
                raise result_box["exc"]
            return

        # Legacy sync path (back-compat for all existing callers).
        tbl = self.db.open_table(RECORDS_TABLE)
        tbl.add([self._to_row(record)])
        # mirror to runtime graph.
        self._fire_graph_sync_hook("insert", record)

    # -------------------------------------------------------- async

    async def enable_async_writes(
        self,
        coalesce_ms: int = 100,
        max_batch: int = 128,
        max_queue_size: int = 4096,
    ) -> None:
        """Switch ``insert()`` onto the coalescing AsyncWriteQueue.

        Runs the queue on a dedicated background event loop so sync
        callers (every existing user of ``store.insert``) can keep
        calling ``insert(record)`` and block on the batch flush via
        ``run_coroutine_threadsafe``. The read path stays synchronous
        and untouched — is owned by .

        Idempotent: a second call while already enabled is a no-op.
        """
        if self._write_queue is not None:
            return

        from iai_mcp.write_queue import AsyncWriteQueue

        # Spawn a dedicated loop on a daemon thread. The calling
        # coroutine stays on the caller's loop — we do not block it.
        ready = threading.Event()
        loop_holder: dict = {}

        def _run() -> None:
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        thread = threading.Thread(
            target=_run, name="iai-mcp-async-writes", daemon=True,
        )
        thread.start()
        ready.wait()
        bg_loop: asyncio.AbstractEventLoop = loop_holder["loop"]

        # Open the async LanceDB connection + table on the background loop.
        async def _open():
            conn = await lancedb.connect_async(str(self.root / "lancedb"))
            tbl = await conn.open_table(RECORDS_TABLE)
            return conn, tbl

        async_conn, async_tbl = asyncio.run_coroutine_threadsafe(
            _open(), bg_loop
        ).result()

        # Adapter: queue enqueues MemoryRecord objects; the real LanceDB
        # tbl.add expects a list of row dicts. We convert here so the
        # queue's on_flushed callback still sees MemoryRecords.
        to_row = self._to_row

        class _RecordTableAdapter:
            def __init__(self, real_tbl) -> None:
                self._real = real_tbl

            async def add(self, records: list) -> None:
                rows = [to_row(r) for r in records]
                await self._real.add(rows)

        adapter = _RecordTableAdapter(async_tbl)

        # on_flushed: fire the 05-12 graph-sync hook once per record in
        # batch order. This is synchronous (runs on the background loop)
        # but the hook itself is pure-python — no blocking I/O expected.
        fire_hook = self._fire_graph_sync_hook

        def _on_flushed(batch: list) -> None:
            for rec in batch:
                fire_hook("insert", rec)

        queue = AsyncWriteQueue(
            adapter,
            coalesce_ms=coalesce_ms,
            max_batch=max_batch,
            max_queue_size=max_queue_size,
            on_flushed=_on_flushed,
        )
        asyncio.run_coroutine_threadsafe(queue.start(), bg_loop).result()

        self._async_loop = bg_loop
        self._async_thread = thread
        self._async_conn = async_conn
        self._write_queue = queue

        # same opt-in enables the provenance queue too.
        # Cleanest ergonomics — anyone who wants async record writes
        # also wants async provenance writes (both are off the
        # user-facing critical path).
        self.enable_provenance_queue()

    async def disable_async_writes(self) -> None:
        """Drain the queue, tear down the background loop.

        After this returns, ``insert()`` reverts to the legacy sync
        path. Idempotent.
        """
        if self._write_queue is None:
            # Still tear down the provenance queue if only that half was up.
            self.disable_provenance_queue()
            return
        # tear down the provenance queue first so in-flight
        # writes drain via the still-live sync append path.
        self.disable_provenance_queue()
        bg_loop = self._async_loop
        queue = self._write_queue
        try:
            asyncio.run_coroutine_threadsafe(queue.stop(), bg_loop).result()
            # Close the async lancedb connection if it exposes close().
            if self._async_conn is not None:
                close = getattr(self._async_conn, "close", None)
                if close is not None:
                    try:
                        maybe = close()
                        if asyncio.iscoroutine(maybe):
                            asyncio.run_coroutine_threadsafe(
                                maybe, bg_loop
                            ).result()
                    except Exception:
                        pass
        finally:
            # Stop the background loop + join its thread.
            if bg_loop is not None:
                bg_loop.call_soon_threadsafe(bg_loop.stop)
            if self._async_thread is not None:
                self._async_thread.join(timeout=5.0)
            self._write_queue = None
            self._async_loop = None
            self._async_thread = None
            self._async_conn = None

    # -------------------------------------------------- provenance queue

    def enable_provenance_queue(self, *, coalesce_ms: int = 50) -> None:
        """route provenance writes through a daemon-thread queue.

        After this call, ``queue_provenance_batch(pairs)`` hands the
        pairs off to a background worker and returns immediately;
        ``pipeline_recall`` no longer blocks on ``append_provenance_batch``.
        Idempotent — a second call with an already-live queue is a
        no-op.

        The queue is purpose-built for provenance (pure side effect,
        never read back). Record inserts still go through the
        ``AsyncWriteQueue`` from ``enable_async_writes()`` because they
        must be durable before return (S4 viability).
        """
        if self._provenance_queue is not None:
            return
        from iai_mcp.provenance_queue import ProvenanceWriteQueue

        q = ProvenanceWriteQueue(self, coalesce_ms=coalesce_ms)
        q.start()
        self._provenance_queue = q

    def disable_provenance_queue(self) -> None:
        """drain + stop the provenance queue.

        After this returns, ``queue_provenance_batch`` reverts to the
        sync fallback. Idempotent.
        """
        q = self._provenance_queue
        if q is None:
            return
        try:
            q.flush(timeout=2.0)
        except Exception:
            pass
        try:
            q.stop()
        except Exception:
            pass
        self._provenance_queue = None

    def queue_provenance_batch(
        self, pairs: "list[tuple[UUID, dict]]"
    ) -> None:
        """fire-and-forget provenance write.

        If the async queue is live, enqueue + return (non-blocking).
        Otherwise fall back to the sync ``append_provenance_batch``
        call — identical behaviour to the pre-05-14 code path. This is
        what ``pipeline_recall`` calls in place of the direct sync write.

        Rule 1: the sync fallback is wrapped in the caller's own
        try/except (pipeline_recall has one); we don't add a second
        layer here so failures surface the same way they always did.
        """
        if not pairs:
            return
        q = self._provenance_queue
        if q is not None:
            q.enqueue(pairs)
            return
        # Sync fallback (back-compat).
        self.append_provenance_batch(pairs, records_cache=None)

    # ------------------------------------------------------- writes

    def update(self, record: MemoryRecord) -> None:
        """full-record update (used by the graph-sync surface).

        Rewrites the core columns we expose on graph node attrs
        (embedding, literal_surface, centrality, tier, pinned) plus
        updated_at. Encrypts literal_surface under the record's AD.
        Missing record id is a silent no-op (matches append_provenance
        semantics). Writes-first, hook-second: store is authoritative.

        Scope note: this is deliberately narrower than _to_row — we only
        touch columns relevant to the runtime recall surface. FSRS-only
        updates should keep using update_record(record). Callers that
        need to rewrite every column (migration path) should delete +
        insert instead.
        """
        if len(record.embedding) != self._embed_dim:
            raise ValueError(
                f"embedding must be {self._embed_dim}d, got {len(record.embedding)}"
            )
        tbl = self.db.open_table(RECORDS_TABLE)
        # Fast existence check before issuing the update.
        df = tbl.to_pandas()
        if df.empty or str(record.id) not in set(df["id"].tolist()):
            return
        literal_ct = self._encrypt_for_record(record.id, record.literal_surface)
        tbl.update(
            where=f"id = '{_uuid_literal(record.id)}'",
            values={
                "literal_surface": literal_ct,
                "embedding": [float(x) for x in record.embedding],
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        # mirror to runtime graph.
        self._fire_graph_sync_hook("update", record)

    def delete(self, record_id: UUID) -> None:
        """remove a record by id + mirror to the runtime graph.

        LanceDB ``tbl.delete(where=...)`` is the authoritative operation.
        Unknown id is a silent no-op. Graph-sync hook fires with a
        minimal shim record carrying only ``id`` so the hook can drop
        the node from the NetworkX graph without needing the full
        payload.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        try:
            tbl.delete(where=f"id = '{_uuid_literal(record_id)}'")
        except Exception:
            # LanceDB raises on malformed WHERE; normalise to no-op so
            # callers get the same semantics as unknown-id.
            return

        # Fire the hook with a minimal shim — the graph sync only needs
        # the id to call G.remove_node.
        class _DeleteShim:
            def __init__(self, rid):
                self.id = rid
        self._fire_graph_sync_hook("delete", _DeleteShim(record_id))

    def get(self, record_id: UUID) -> MemoryRecord | None:
        """/ filter-pushdown point read.

        Replaces the legacy O(N) ``tbl.to_pandas()`` full-scan (which
        materialised every row + every column into pandas and then
        filtered in-process -- ~34 ms/call on the prod schema at
        N=1k, ~340 ms per recall iteration across the L0 fast-path +
        anti-hit lookup) with a LanceDB filter-pushdown point read via
        ``tbl.search().where(...).limit(1).to_pandas()``. Lance pushes
        the predicate into the scanner so only the matching row is
        materialised; cost becomes O(index-lookup), sub-ms at N=1k.

        Semantics preserved exactly: unknown id -> None;
        existing id -> ``MemoryRecord`` via ``_from_row`` (AES-GCM
        decrypt path untouched). ``_uuid_literal`` gates the predicate
        against SQL-injection / malformed-UUID inputs.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        df = (
            tbl.search()
            .where(f"id = '{_uuid_literal(record_id)}'")
            .limit(1)
            .to_pandas()
        )
        if df.empty:
            return None
        return self._from_row(df.iloc[0].to_dict())

    def all_records(self) -> list[MemoryRecord]:
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        return [self._from_row(r.to_dict()) for _, r in df.iterrows()]

    # (..): streaming + projection — see internal architecture spec
    def iter_records(
        self,
        *,
        columns: list[str] | None = None,
        batch_size: int = 1024,
        where: str | None = None,
    ):
        """W1: streaming + projection iterator over records.

        Yields ``MemoryRecord`` instances batch by batch via LanceDB's
        documented memory-efficient pattern. Unlike :meth:`all_records`,
        nothing is materialised into a single in-memory list; downstream
        consumers (sleep daemon, S4 scans) can process records lazily and
        keep peak RSS bounded.

        Parameters
        ----------
        columns:
            If given, only these columns are read from disk. Encrypted
            columns NOT in this list are never decrypted (zero AES-GCM cost
            for the projected read). When ``None``, all columns are read
            (parity with :meth:`all_records`).
        batch_size:
            Rows per LanceDB ``RecordBatch``. Default 1024 -- small enough
            that 384d-embedding rows fit comfortably in working set, large
            enough that scanner overhead is amortised.
        where:
            Optional SQL-style predicate forwarded to LanceDB's scanner.
            Example: ``"tier = 'episodic'"``. ``None`` = full scan.

        Notes
        -----
        Surface is ``tbl.search().where(...).select([...]).to_batches(batch_size=N)``;
        on lancedb 0.30.2 the alternative ``tbl.to_lance().to_batches(...)``
        raises ``ImportError`` because the optional ``pylance`` extra is not
        installed.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        query = tbl.search()
        if where is not None:
            query = query.where(where)
        if columns is not None:
            query = query.select(columns)
        reader = query.to_batches(batch_size=batch_size)
        for batch in reader:
            for row_dict in batch.to_pylist():
                yield self._from_row(row_dict)

    def iter_record_columns(
        self,
        columns: list[str],
        *,
        batch_size: int = 1024,
        where: str | None = None,
    ):
        """W2: projection-only iteration; no MemoryRecord, no decrypt.

        Yields raw ``dict`` rows containing only the requested columns. Encrypted
        fields (literal_surface, provenance_json, profile_modulation_gain_json),
        if listed in ``columns``, pass through as ciphertext strings -- the caller
        decides whether to decrypt. For tag-only paths like
        :func:`iai_mcp.sleep._tier0_schema_surfacing` projecting ``["tags_json"]``,
        no AES-GCM operations happen anywhere on the path.

        Parameters mirror :meth:`iter_records`. ``columns`` is REQUIRED -- this
        method exists specifically for projection-only iteration; if you want
        every column, use :meth:`all_records` or :meth:`iter_records`.
        """
        if not columns:
            raise ValueError("iter_record_columns requires a non-empty columns list")
        tbl = self.db.open_table(RECORDS_TABLE)
        query = tbl.search()
        if where is not None:
            query = query.where(where)
        query = query.select(columns)
        reader = query.to_batches(batch_size=batch_size)
        for batch in reader:
            for row_dict in batch.to_pylist():
                yield row_dict

    def query_similar(
        self,
        vec: list[float],
        k: int = 10,
        tier: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine-distance kNN search. Returns (record, cosine_similarity) pairs.

        LanceDB's default L2 distance is mapped via explicit `.distance_type("cosine")`
        so `_distance` is cosine distance; we return `1.0 - _distance` as similarity.

        / optional ``tier`` kwarg applies a LanceDB
        where-clause filter at the search layer. Validated against the
        canonical ``TIER_ENUM`` (imported from ``iai_mcp.types``); bad
        tier values raise ``ValueError`` BEFORE any I/O is attempted, so
        the validation also acts as a SQL-injection guard for the string
        interpolation below (tier values are alphanumeric ASCII so direct
        interpolation is safe once the validation succeeds). When
        ``tier=None``, behaviour is byte-identical to the legacy zero-tier
        contract -- no where-clause is appended.
        """
        # step 1: validate `tier` BEFORE any I/O so a bad value never
        # touches LanceDB. Sentinel raise lets callers (capture_turn) catch
        # ValueError specifically on the bad-tier path.
        if tier is not None and tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {tier!r}; must be one of {sorted(TIER_ENUM)}"
            )

        tbl = self.db.open_table(RECORDS_TABLE)
        # Fast path for empty store -- tbl.search on empty raises or returns empty;
        # the explicit check also avoids LanceDB warnings about missing indices at N=0.
        if tbl.count_rows() == 0:
            return []
        # Build the query chain. Mirrors the predicate-where idiom at
        # `iter_records` (lines 930-935 of this file).
        q = tbl.search(list(vec)).distance_type("cosine")
        if tier is not None:
            # Tier validated above against TIER_ENUM (alphanumeric ASCII), so
            # direct string interpolation here is safe.
            q = q.where(f"tier = '{tier}'")
        results = q.limit(k).to_pandas()
        out: list[tuple[MemoryRecord, float]] = []
        for _, row in results.iterrows():
            record = self._from_row(row.to_dict())
            # LanceDB returns `_distance` as cosine distance in [0, 2]; similarity = 1 - distance.
            distance = float(row.get("_distance", 1.0)) if "_distance" in row else 1.0
            score = 1.0 - distance
            out.append((record, score))
        return out

    def update_record(self, record: MemoryRecord) -> None:
        """H-01: persist FSRS-relevant columns back to the records table.

        Scope (deliberately narrow):
            stability, difficulty, last_reviewed, updated_at

        Everything else on the record (embedding, provenance_json, tags_json,
        community_id, centrality, structure_hv, schema_version, language,
        s5_trust_score, profile_modulation_gain_json) is LEFT UNTOUCHED so this
        method cannot clobber concurrent writers (boost_edges / append_provenance
        / migrate_v1_to_v2). LanceDB's tbl.update(values=...) only rewrites the
        listed columns.

        Unknown record id is a silent no-op (no exception, no table growth) --
        matches append_provenance semantics.

        H-01 bug: run_light_consolidation's _apply_fsrs mutated the in-memory
        MemoryRecord but never wrote it back; every process restart reset
        stability + last_reviewed to their last-persisted value. This method
        closes that gap.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        idx = df.index[df["id"] == str(record.id)].tolist()
        if not idx:
            return
        tbl.update(
            where=f"id = '{_uuid_literal(record.id)}'",
            values={
                "stability": float(record.stability),
                "difficulty": float(record.difficulty),
                "last_reviewed": record.last_reviewed,
                "updated_at": datetime.now(timezone.utc),
            },
        )

    # -------------------------------------------------------- reconsolidation

    def append_provenance(self, record_id: UUID, entry: dict) -> None:
        """append a provenance entry to the record.

        Read-modify-write per (sync write, acceptable for single-user ).
        existing provenance is decrypted when encrypted; the
        updated list is re-encrypted before write.
        """
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        idx = df.index[df["id"] == str(record_id)].tolist()
        if not idx:
            return
        i = idx[0]
        raw = df.at[i, "provenance_json"] or "[]"
        if is_encrypted(raw):
            raw = self._decrypt_for_record(record_id, raw)
        try:
            existing = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            existing = []
        existing.append(entry)
        new_json_plain = json.dumps(existing)
        new_json_ct = self._encrypt_for_record(record_id, new_json_plain)
        tbl.update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={
                "provenance_json": new_json_ct,
                "updated_at": datetime.now(timezone.utc),
            },
        )

    def append_provenance_batch(
        self, pairs: "list[tuple[UUID, dict]]",
        records_cache: "dict | None" = None,
    ) -> None:
        """: batched provenance append ( preserved).

        Collapses the per-hit N+1 `to_pandas()` scan pattern into:
          * ONE `tbl.to_pandas()` scan to read current provenance (or ZERO
            if `records_cache` is provided -- we read provenance from the
            fresh cache pipeline_recall already built).
          * ONE `tbl.merge_insert(...)` transaction to write back all updates
            in a single LanceDB operation. This replaces O(unique_ids)
            separate `tbl.update()` calls that each cost ~13ms on real
            hardware (11 hits x 13ms = 143ms before; single merge_insert
            ~10ms after = ~14x faster).

        Semantics match append_provenance:
        - Each entry is appended to its record's provenance list (concat, not replace).
        - Unknown record_ids are silently skipped.
        - Empty `pairs` -> no-op.
        - Order of entries per record is preserved (same order they appear in
          `pairs`); this matches N individual append_provenance calls on the
          same pairs.
        - `updated_at` is bumped once per unique record id (matching single-call
          semantics; the exact timestamp value differs from N individual calls
          but that is expected and excluded from equivalence tests).
        - `merge_insert` with a subset of columns ('id', 'provenance_json',
          'updated_at') preserves all other columns untouched (embedding,
          tags_json, aaak_index, etc.) -- same guarantee as the single-call
          `tbl.update(values={...})` surface.

        Why this is the perf-critical surface ( SC-6):
        Pre-fix: pipeline_recall -> for hit in hits: store.append_provenance(...)
                 => N x to_pandas() scans (~20ms each, dominant cost at N=5-11).
        Post-fix: pipeline_recall -> store.append_provenance_batch([...],
                                    records_cache=records_cache)
                 => 0 x to_pandas() scans + 1 x merge_insert transaction.

        records_cache: optional dict[UUID | str, MemoryRecord]. When provided,
        existing provenance is read from the cache's MemoryRecord.provenance
        list (already deserialised) -- skipping the full-table scan entirely.
        The cache must have been loaded recently (pipeline_recall builds it
        at stage 1, then calls this method before any other mutation). If the
        cache is missing an id, that id is silently skipped (matches
        single-call unknown-id semantics).
        """
        if not pairs:
            return
        tbl = self.db.open_table(RECORDS_TABLE)

        # Group entries by record_id, preserving per-record insertion order.
        from collections import defaultdict
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rid, entry in pairs:
            grouped[str(rid)].append(entry)

        # Build the merge-insert payload: one row per unique id with the new
        # provenance_json (existing list + appended entries) and fresh updated_at.
        now = datetime.now(timezone.utc)
        update_ids: list[str] = []
        update_prov: list[str] = []

        if records_cache is not None:
            # Fast path: read existing provenance from the pre-loaded cache.
            # Zero scan. Keyed by UUID object OR str (be permissive).
            for rid_str, entries in grouped.items():
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                # Try UUID object key first, then str fallback.
                try:
                    rec = records_cache.get(UUID(rid_str))
                except (TypeError, ValueError):
                    rec = None
                if rec is None:
                    rec = records_cache.get(rid_str)
                if rec is None:
                    # Not in cache -- silently skip (matches single-call semantics).
                    continue
                existing = list(rec.provenance or [])
                existing.extend(entries)
                # encrypt the new provenance JSON so the updated row
                # matches the encrypted contract enforced by insert().
                new_plain = json.dumps(existing)
                new_ct = self._encrypt_for_record(UUID(rid_str), new_plain)
                update_ids.append(canonical)
                update_prov.append(new_ct)
        else:
            # Slow path: one full to_pandas() scan for existing provenance.
            df = tbl.to_pandas()
            if df.empty:
                return
            for rid_str, entries in grouped.items():
                idx_list = df.index[df["id"] == rid_str].tolist()
                if not idx_list:
                    continue
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                i = idx_list[0]
                raw_prov = df.at[i, "provenance_json"] or "[]"
                # decrypt pre-existing ciphertext before merging
                # (fresh entries are plaintext dicts).
                if is_encrypted(raw_prov):
                    try:
                        raw_prov = self._decrypt_for_record(UUID(rid_str), raw_prov)
                    except Exception:
                        raw_prov = "[]"
                try:
                    existing = json.loads(raw_prov)
                except (TypeError, ValueError):
                    existing = []
                existing.extend(entries)
                new_plain = json.dumps(existing)
                new_ct = self._encrypt_for_record(UUID(rid_str), new_plain)
                update_ids.append(canonical)
                update_prov.append(new_ct)

        if not update_ids:
            return

        # Single merge_insert transaction: join on `id`, update matched rows'
        # provenance_json + updated_at columns. All other record columns are
        # preserved untouched (merge_insert with subset columns is surgical).
        import pyarrow as pa
        update_tbl = pa.table({
            "id": update_ids,
            "provenance_json": update_prov,
            "updated_at": [now] * len(update_ids),
        })
        try:
            tbl.merge_insert("id").when_matched_update_all().execute(update_tbl)
        except Exception:
            # Rule 1: never block recall on a provenance-write failure.
            # Fallback: per-id tbl.update() (slower but correct).
            for rid_str, new_json in zip(update_ids, update_prov):
                try:
                    tbl.update(
                        where=f"id = '{rid_str}'",
                        values={
                            "provenance_json": new_json,
                            "updated_at": now,
                        },
                    )
                except Exception:
                    continue

    # ------------------------------------------------------------------ edges

    def boost_edges(
        self,
        pairs: list[tuple[UUID, UUID]],
        delta: float | Sequence[float] = 0.1,
        edge_type: str = "hebbian",
    ) -> dict[tuple[str, str], float]:
        """ + edge-type extension: pairwise edge boost.

        accepts any edge_type from EDGE_TYPES (8 values):
        {hebbian, contradicts, consolidated_from, schema_instance_of,
         temporal_next, invariant_anchor, curiosity_bridge, profile_modulates}.

        Edge key is canonicalised to sorted (src, dst) so (a, b) and (b, a) collide.
        Returns the new weight for each pair (tuple keys).

        refactor: produces AT MOST 2 LanceDB versions per call (one for
        `merge_insert` updating pre-existing rows, one for `tbl.add` of new rows)
        regardless of pair count. Previously each pair issued its own
        `tbl.update`/`tbl.add` plus a per-pair `tbl.to_pandas()` refresh
        (N+1 scans + N versions per call). Today's path:

        1. Validate `edge_type` and coerce `delta` to a per-pair list.
        2. Coalesce duplicate canonical (src, dst) keys IN-MEMORY by summing
           their deltas (preserves the legacy semantic that
           `[(a,b), (a,b)]` with `delta=0.1` accumulates to `cur + 0.2`).
        3. ONE `tbl.to_pandas()` to load existing edges.
        4. Partition into update_rows (key already present) and insert_rows.
        5. ONE `tbl.merge_insert(["src","dst","edge_type"]).when_matched_update_all().execute(arrow)`
           for updates (composite-key merge_insert verified on LanceDB 0.30.2).
        6. ONE `tbl.add(insert_rows)` for new rows.
        7. Returns `dict[tuple[str, str], float]` keyed by canonical sorted (src, dst).

        `delta` accepts a scalar (applied to every pair, backwards-compatible) or
        a `Sequence[float]` of per-pair deltas. Length mismatch raises
        `ValueError`. Used by `pipeline.recall_hook` for per-hit profile gains.
        """
        if edge_type not in EDGE_TYPES:
            raise ValueError(
                f"invalid edge_type {edge_type!r}; must be one of {sorted(EDGE_TYPES)}"
            )

        # Coerce delta to per-pair list. Length validation BEFORE any work.
        if isinstance(delta, (int, float)):
            deltas = [float(delta)] * len(pairs)
        else:
            deltas = [float(d) for d in delta]
            if len(deltas) != len(pairs):
                raise ValueError(
                    f"deltas length {len(deltas)} != pairs length {len(pairs)}"
                )

        if not pairs:
            return {}

        # Coalesce duplicate canonical (src, dst) keys IN-MEMORY: SUM their
        # deltas. A7 acceptance: `[(a,b), (a,b)]` with delta=0.1 -> cur + 0.2,
        # NOT cur + 0.1. The legacy per-pair tbl.to_pandas() refresh existed
        # purely to support this semantic; in-memory coalescing replaces it.
        coalesced: dict[tuple[str, str], float] = {}
        for (a, b), d in zip(pairs, deltas):
            key = (str(a), str(b))
            canonical = tuple(sorted(key))
            coalesced[canonical] = coalesced.get(canonical, 0.0) + d
        if not coalesced:
            return {}

        tbl = self.db.open_table(EDGES_TABLE)

        # ONE full-table scan at entry. Acceptable at the project's edge-count
        # scale (<= ~5K rows). A scoped `tbl.search().where(...)` predicate is
        # a follow-up micro-optimisation per CONTEXT .
        existing = tbl.to_pandas()

        update_rows: list[dict] = []
        insert_rows: list[dict] = []
        new_weights: dict[tuple[str, str], float] = {}
        now = datetime.now(timezone.utc)

        for (src_str, dst_str), accum_delta in coalesced.items():
            if len(existing) > 0:
                mask = (
                    (existing["src"] == src_str)
                    & (existing["dst"] == dst_str)
                    & (existing["edge_type"] == edge_type)
                )
            else:
                mask = None
            if mask is not None and mask.any():
                cur = float(existing.loc[mask, "weight"].iloc[0])
                nw = cur + accum_delta
                update_rows.append(
                    {
                        "src": src_str,
                        "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw,
                        "updated_at": now,
                    }
                )
            else:
                nw = accum_delta
                insert_rows.append(
                    {
                        "src": src_str,
                        "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw,
                        "updated_at": now,
                    }
                )
            new_weights[(src_str, dst_str)] = nw

        # ONE merge_insert for updates. Composite key (src, dst, edge_type) is
        # verified working on LanceDB 0.30.2 (probe in RESEARCH F-5).
        # Fallback to per-row tbl.update preserves correctness on any future
        # LanceDB regression.
        if update_rows:
            try:
                upd_arrow = pa.Table.from_pylist(
                    update_rows,
                    schema=pa.schema(
                        [
                            ("src", pa.string()),
                            ("dst", pa.string()),
                            ("edge_type", pa.string()),
                            ("weight", pa.float32()),
                            ("updated_at", pa.timestamp("us", tz="UTC")),
                        ]
                    ),
                )
                (
                    tbl.merge_insert(["src", "dst", "edge_type"])
                    .when_matched_update_all()
                    .execute(upd_arrow)
                )
            except Exception:
                # Fallback: per-row tbl.update. Slower (N versions) but
                # correctness-preserving if merge_insert ever misbehaves.
                for r in update_rows:
                    tbl.update(
                        where=(
                            f"src = '{_uuid_literal(r['src'])}' "
                            f"AND dst = '{_uuid_literal(r['dst'])}' "
                            f"AND edge_type = '{edge_type}'"
                        ),
                        values={
                            "weight": r["weight"],
                            "updated_at": r["updated_at"],
                        },
                    )

        # ONE tbl.add for new rows.
        if insert_rows:
            tbl.add(insert_rows)

        return new_weights

    def reinforce_record(
        self,
        record_id: UUID,
        anchor_id: UUID | None = None,
        edge_type: str = "hebbian",
        delta: float = 0.1,
    ) -> dict[tuple[str, str], float]:
        """ typed wrapper: single-record Hebbian reinforcement.

        / step 2 — the canonical reinforcement target for
        ``memory_capture`` dedup-on-cos>=0.95. Promoting this typed wrapper
        next to ``boost_edges`` makes the single-record-reinforcement intent
        explicit at the call site and prevents the Bug-C shape-mismatch
        (single-UUID list passed to a tuple-of-pairs API) from recurring.

        When ``anchor_id is None`` (the dedup-call shape), this records a
        ``(record_id, record_id)`` self-loop edge — the canonical self-loop
        semantic for ``capture_turn``'s dedup path. Self-loop is chosen over
        a record-counter because it reuses every line of ``boost_edges``
        and the canonical-pair coalescer at boost_edges:1244-1247 produces
        the right key shape with no schema or table changes.

        When ``anchor_id`` is provided, routes to the existing pair-mode
        contract (``anchor_id`` -> ``record_id``) edge — preserves the
        legacy two-record reinforcement semantics for callers that already
        had an anchor.

        Returns the same ``dict[tuple[str, str], float]`` shape as
        :meth:`boost_edges`. ``edge_type`` validation is delegated to
        ``boost_edges`` (the existing ``EDGE_TYPES`` check at lines
        1221-1224); a second validation here would be redundant — one
        source of truth.

        See PATTERNS.md store.py Analog 4 for the precedent
        (``hebbian_structure.strengthen_structure_edge``), which is the
        same shape: a thin typed wrapper that builds a single-pair list
        and delegates to ``boost_edges``.
        """
        if anchor_id is None:
            pair = (record_id, record_id)
        else:
            pair = (anchor_id, record_id)
        return self.boost_edges([pair], delta=delta, edge_type=edge_type)

    def add_contradicts_edge(self, original: UUID, new_id: UUID) -> None:
        """ edge-based reconsolidation: original unchanged."""
        tbl = self.db.open_table(EDGES_TABLE)
        tbl.add(
            [
                {
                    "src": str(original),
                    "dst": str(new_id),
                    "edge_type": "contradicts",
                    "weight": 1.0,
                    "updated_at": datetime.now(timezone.utc),
                }
            ]
        )

    # ---------------------------------------------------------------- helpers

    def _to_row(self, r: MemoryRecord) -> dict:
        # encrypt sensitive columns with AD = record.id.
        # literal_surface, provenance_json, profile_modulation_gain_json
        # are the three encrypted columns on the records table.
        literal_ct = self._encrypt_for_record(r.id, r.literal_surface)
        provenance_plain = json.dumps(r.provenance)
        provenance_ct = self._encrypt_for_record(r.id, provenance_plain)
        gain_plain = json.dumps(r.profile_modulation_gain or {})
        gain_ct = self._encrypt_for_record(r.id, gain_plain)
        return {
            "id": str(r.id),
            "tier": r.tier,
            "literal_surface": literal_ct,
            "aaak_index": r.aaak_index,
            "embedding": [float(x) for x in r.embedding],
            # : structure_hv is raw bytes (D=10000 BSC packed
            # to 1250 bytes). Empty bytes default for pre-migration / lazy bind.
            "structure_hv": bytes(r.structure_hv or b""),
            "community_id": str(r.community_id) if r.community_id else "",
            "centrality": float(r.centrality),
            "detail_level": int(r.detail_level),
            "pinned": bool(r.pinned),
            "stability": float(r.stability),
            "difficulty": float(r.difficulty),
            "last_reviewed": r.last_reviewed,
            "never_decay": bool(r.never_decay),
            "never_merge": bool(r.never_merge),
            "provenance_json": provenance_ct,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "tags_json": json.dumps(r.tags),
            # v2 columns
            "language": str(r.language),
            "s5_trust_score": float(r.s5_trust_score),
            "profile_modulation_gain_json": gain_ct,
            "schema_version": int(r.schema_version),
        }

    def _from_row(self, row: dict) -> MemoryRecord:
        from uuid import UUID as _UUID

        import pandas as pd  # local import: only hot on reads

        # partial-row safety. iter_records consumers may
        # project a subset of columns; any non-projected column is absent from
        # the row dict. `id` is the primary key and projection without it is a
        # caller bug, not a graceful-fallback case -- fail loud.
        if "id" not in row:
            raise KeyError(
                "iter_records consumer must include 'id' in column projection"
            )

        # read path: prefer the v4 `structure_hv` (pa.binary)
        # column. Legacy v3 stores still expose the old `hd_vector_json` column
        # until migrate_hd_vector_to_structure_hv_v3_to_v4 has run; in that case
        # we surface b"" so MemoryRecord stays valid (the column carried JSON
        # `null` / "" in -- it was never populated).
        structure_raw = row.get("structure_hv")
        if structure_raw is None:
            structure_hv = b""
        elif isinstance(structure_raw, (bytes, bytearray)):
            structure_hv = bytes(structure_raw)
        else:
            structure_hv = b""

        community_raw = row.get("community_id") or ""
        community_id = _UUID(community_raw) if community_raw else None

        # Back-compat read path: a v1 row (or externally written row) may
        # lack language/s5_trust_score/profile_modulation_gain_json/schema_version.
        # Fill with Phase-1 defaults: language="en", s5=0.5, gain={}, version=1.
        #
        # migration note: for schema_version=1 rows with empty language,
        # we preserve the empty string on the in-memory record so migrate_v1_to_v2
        # can run langdetect. MemoryRecord.__post_init__ requires non-empty
        # language, so we pass a placeholder and then null it back out before
        # returning. For v2 rows (or anything missing a schema_version) we
        # default to "en" as before -- those paths don't run migration.
        lang_raw = row.get("language")
        raw_version = row.get("schema_version")
        try:
            version_int = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
        except (TypeError, ValueError):
            version_int = SCHEMA_VERSION_CURRENT
        schema_version = version_int

        is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
        if is_empty_language and schema_version == 1:
            # v1 legacy row -> preserve empty so migration can re-detect.
            # We use a placeholder to satisfy __post_init__ then reset below.
            language = "__LEGACY_EMPTY__"
        elif is_empty_language:
            language = "en"
        else:
            language = str(lang_raw)

        s5_raw = row.get("s5_trust_score")
        s5_trust_score = float(s5_raw) if s5_raw is not None else 0.5

        # decrypt profile_modulation_gain_json if it carries the
        # iai:enc:v1: prefix (mixed plaintext/ciphertext during v2->v3 migration).
        from uuid import UUID as _UUID2
        _row_uuid = _UUID2(row["id"])
        gain_raw = row.get("profile_modulation_gain_json") or "{}"
        if is_encrypted(gain_raw):
            gain_raw = self._decrypt_for_record(_row_uuid, gain_raw)
        try:
            profile_modulation_gain = json.loads(gain_raw) or {}
        except (TypeError, json.JSONDecodeError):
            profile_modulation_gain = {}

        # Pandas sentinel -> None normalisation: LanceDB returns NaT for null
        # timestamp columns. NaT doesn't round-trip back through PyArrow on
        # insert (migrate_v1_to_v2 depends on this). Coerce NaT -> None so the
        # MemoryRecord's last_reviewed is cleanly None.
        last_reviewed_raw = row.get("last_reviewed")
        try:
            last_reviewed = None if pd.isna(last_reviewed_raw) else last_reviewed_raw
        except (TypeError, ValueError):
            last_reviewed = last_reviewed_raw

        # decrypt literal_surface + provenance_json if encrypted.
        # bracket access hardened to defensive .get() so
        # column-projected reads (where these columns may be absent) do not
        # KeyError. is_encrypted("") and is_encrypted("[]") are both False,
        # so the empty-default flows through as plaintext untouched.
        row_uuid = _UUID(row["id"])
        literal_raw = row.get("literal_surface", "")
        if is_encrypted(literal_raw):
            literal_raw = self._decrypt_for_record(row_uuid, literal_raw)
        provenance_raw = row.get("provenance_json") or "[]"
        if is_encrypted(provenance_raw):
            provenance_raw = self._decrypt_for_record(row_uuid, provenance_raw)
        try:
            provenance_list = json.loads(provenance_raw) if provenance_raw else []
        except (TypeError, json.JSONDecodeError):
            provenance_list = []

        rec = MemoryRecord(
            id=row_uuid,
            tier=row.get("tier", "episodic"),
            literal_surface=literal_raw,
            aaak_index=row.get("aaak_index") or "",
            embedding=(
                list(row["embedding"])
                if row.get("embedding") is not None
                else []
            ),
            community_id=community_id,
            centrality=float(row.get("centrality", 0.0) or 0.0),
            detail_level=int(row.get("detail_level", 1)),
            pinned=bool(row.get("pinned", False)),
            stability=float(row.get("stability") or 0.0),
            difficulty=float(row.get("difficulty") or 0.0),
            last_reviewed=last_reviewed,
            never_decay=bool(row.get("never_decay", False)),
            never_merge=bool(row.get("never_merge", False)),
            provenance=provenance_list,
            created_at=row.get("created_at") or datetime.now(timezone.utc),
            updated_at=row.get("updated_at") or datetime.now(timezone.utc),
            tags=json.loads(row.get("tags_json") or "[]"),
            language=language,
            s5_trust_score=s5_trust_score,
            profile_modulation_gain=profile_modulation_gain,
            schema_version=schema_version,
            structure_hv=structure_hv,
        )
        if language == "__LEGACY_EMPTY__":
            rec.language = ""  # post-construction: signal to migration path
        return rec
