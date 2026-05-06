"""Phase 07.11 Plan 02 / — empty-surface cache poisoning regression.

AES-GCM decrypt failure during
graph build poisons the runtime-graph cache with empty `surface` strings,
and on warm-restart the poisoned cache rehydrates as if those records had
genuinely empty literals. The pipeline read path then returns "" claiming
success, violating the verbatim-recall invariant.

Three coordinated changes:

1. `retrieve.py` graph-build decrypt error handler must NOT write empty
   surface to the live NetworkX graph OR to ``node_payload_for_cache``.
   Skip the row entirely + emit a structured ``graph_build_decrypt_failed``
   log event.
2. `pipeline._read_record_payload` treats empty/None surface OR
   ``_decrypt_failed=True`` as a cache miss and falls back to ``store.get``.
3. `runtime_graph_cache.try_load`, on rehydrate, drops poisoned entries
   (surface in (None, "") OR ``_decrypt_failed`` flag set) and emits a
   structured ``runtime_graph_cache_drop_poisoned_entry`` stderr event.

Three regression tests, one per coordinated change. Each test fails on
``git stash`` of the source diffs (RED witness) and passes with the fix
applied (GREEN witness).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    """Project-canonical isolated-keyring fixture (mirrors
    test_pipeline_anti_hits_malformed.py:33-50). Without this the test
    hangs on macOS keychain GUI prompts on the construction host."""
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """Fresh MemoryStore in tmp_path/lancedb with cache file under tmp_path."""
    s = MemoryStore(path=tmp_path / "lancedb")
    # Override root so the cache file lands at tmp_path/runtime_graph_cache.json
    # (mirrors the pattern in test_runtime_graph_cache.py:60 +
    # test_data_integrity_soak.py:207).
    s.root = tmp_path
    return s


def _make_record(rid: UUID, surface: str = "topic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _write_encrypted_cache(store: MemoryStore, path: Path, data: dict) -> None:
    """Encrypt and write a hand-modified JSON dict to the v3 ciphertext
    sidecar. Mirrors tests/test_runtime_graph_cache.py:82-93 verbatim."""
    from iai_mcp import runtime_graph_cache
    from iai_mcp.crypto import encrypt_field

    plaintext = json.dumps(data)
    ciphertext = encrypt_field(
        plaintext,
        store._key(),
        runtime_graph_cache._CACHE_AAD,
    )
    path.write_text(ciphertext, encoding="ascii")


# --------------------------------------------------------------------------- case A


def test_decrypt_failure_skips_cache_write(store, tmp_path):
    """D-02 step 1: a tampered ciphertext on one record must NOT poison
    the runtime-graph cache. The poisoned record's id MUST be absent
    from the on-disk cache's ``node_payload`` after build_runtime_graph
    runs through the decrypt-failure path.

    AD-tamper trick (mirrors tests/test_store_encrypted.py:250-274):
    overwrite r_b's literal_surface column with r_a's ciphertext. The AD
    (associated data) bound to that ciphertext is r_a.id, so the read
    path's `_decrypt_for_record(r_b.id, ...)` call raises (AAD mismatch
    fails the GCM tag check).

    Without the fix retrieve.py writes ``surface=""`` to
    ``node_payload_for_cache[str(r_b.id)]``, then runtime_graph_cache.save
    persists that empty-surface entry. With the fix r_b.id is skipped
    entirely from the cache.
    """
    from iai_mcp import retrieve, runtime_graph_cache
    from iai_mcp.store import RECORDS_TABLE, _uuid_literal

    # Two clean records.
    r_a = _make_record(uuid4(), "row A — clean surface")
    r_b = _make_record(uuid4(), "row B — to be tampered")
    store.insert(r_a)
    store.insert(r_b)

    # Read both rows' literal_surface ciphertexts.
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    ct_a = df[df["id"] == str(r_a.id)].iloc[0]["literal_surface"]

    # AD-tamper: overwrite row B's literal_surface with row A's
    # ciphertext. The AD bound to ct_a is r_a.id; decrypting against
    # r_b.id will fail tag verification.
    tbl.update(
        where=f"id = '{_uuid_literal(r_b.id)}'",
        values={"literal_surface": ct_a},
    )

    # Build the runtime graph. retrieve.py's decrypt-fail path (post-fix)
    # must skip r_b entirely and NOT write ``surface=""`` to the cache.
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    # build_runtime_graph already calls runtime_graph_cache.save on
    # cache-miss paths. Reload from disk to inspect the persisted shape.
    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None, "cache should have been persisted by build_runtime_graph"
    _, _, payload, _ = loaded

    # Clean record's id is in the cache; poisoned record's id is NOT.
    assert str(r_a.id) in payload, "clean record must be cached"
    assert str(r_b.id) not in payload, (
        "poisoned record (decrypt-fail) must NOT be in the cache — "
        "an empty surface there is the V2-03 poisoning bug"
    )


# --------------------------------------------------------------------------- case B


def test_pipeline_falls_back_to_store_on_empty_surface(store, tmp_path):
    """D-02 step 2: synthetically poison the live graph node payload
    (set surface=""). pipeline._read_record_payload must round-trip via
    store.get(rid) and return the original literal_surface, NOT the
    poisoned empty string.

    Without the fix the function reads the empty surface directly from
    the graph node and returns ``literal_surface=""`` — silent corruption
    of verbatim recall. With the fix empty surface is treated as
    a cache-miss sentinel and store.get fills in the canonical value.
    """
    from iai_mcp import pipeline, retrieve

    rid = uuid4()
    original = "the literal surface that must round-trip"
    store.insert(_make_record(rid, original))

    graph, _assignment, _rich_club = retrieve.build_runtime_graph(store)

    # Synthetic poison: zero out the live node's surface to simulate
    # a poisoned cache rehydrate. (We do NOT use the AD-tamper trick
    # here because Task 1 fixes retrieve.py to skip the node entirely
    # on tamper — the tamper would mean the node isn't in the graph at
    # all, defeating the purpose of testing pipeline's empty-surface
    # guard. The synthetic poison is the orthogonal regression target.)
    assert str(rid) in graph._nx.nodes, "node should exist post-build"
    graph._nx.nodes[str(rid)]["surface"] = ""

    # _read_record_payload takes a NetworkX graph (G.nodes.get); pipeline
    # callers always pass `graph._nx` (cf. pipeline.py:717 `G = graph._nx`).
    out = pipeline._read_record_payload(graph._nx, rid, store)
    assert out is not None, "store.get fallback must produce a record"
    # The returned object's literal_surface must equal the original,
    # round-tripped via store.get's decrypt path. Field name varies
    # by return type (SimpleRecordView vs MemoryRecord) but both expose
    # `literal_surface`.
    assert out.literal_surface == original, (
        "empty-surface graph node must fall through to store.get; "
        "instead got "
        f"{out.literal_surface!r}"
    )


# --------------------------------------------------------------------------- case C


def test_runtime_graph_cache_drops_poisoned_entries_on_load(
    store, tmp_path, capsys
):
    """D-02 step 3: hand-write an encrypted cache containing one good
    node and one poisoned (surface="") node. try_load must drop the
    poisoned entry and emit a runtime_graph_cache_drop_poisoned_entry
    stderr event.

    Construct the OUTER cache shape exactly the way runtime_graph_cache.save
    writes it (cache_version, key, assignment, rich_club, node_payload,
    max_degree, saved_at). The OUTER decode path (assignment +
    rich_club + key match against current store state) must succeed so
    that the INNER node_payload filter is exercised.
    """
    from iai_mcp import runtime_graph_cache

    # Insert one record so the outer key (records_count, edges_count,
    # schema_version, embed_dim, cache_version) matches what try_load
    # computes against the live store. Records_count = 1 → cache key
    # tail must reflect 1.
    rid_real = uuid4()
    store.insert(_make_record(rid_real, "real record present in lancedb"))

    good_id = uuid4()
    bad_id = uuid4()

    # OUTER shape: minimum viable assignment + rich_club that
    # _decode_assignment / _decode_rich_club accept. Empty node_to_community
    # / centroids / mid_regions are valid; modularity is a float; backend
    # is a string; top_communities is a list of UUID-strs.
    data = {
        "cache_version": runtime_graph_cache.CACHE_VERSION,
        "key": list(runtime_graph_cache._cache_key(store)),
        "assignment": {
            "node_to_community": {},
            "community_centroids": {},
            "modularity": 0.0,
            "backend": "leiden-test",
            "top_communities": [],
            "mid_regions": {},
        },
        "rich_club": [],
        "node_payload": {
            str(good_id): {
                "embedding": [0.1] * EMBED_DIM,
                "surface": "good record",
                "centrality": 0.0,
                "tier": "episodic",
                "pinned": False,
                "tags": [],
                "language": "en",
            },
            str(bad_id): {
                "embedding": [0.2] * EMBED_DIM,
                "surface": "",  # POISONED — must be dropped on rehydrate
                "centrality": 0.0,
                "tier": "episodic",
                "pinned": False,
                "tags": [],
                "language": "en",
            },
        },
        "max_degree": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_path = tmp_path / "runtime_graph_cache.json"
    _write_encrypted_cache(store, cache_path, data)

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None, (
        "outer decode must succeed; if this fails the fixture is wrong, "
        "not the production code"
    )
    _assignment, _rich_club, payload, _max_degree = loaded

    assert payload is not None
    assert str(good_id) in payload, "well-formed entry must survive rehydrate"
    assert str(bad_id) not in payload, (
        "poisoned (surface='') entry must be dropped — that is V2-03 fix"
    )

    captured = capsys.readouterr()
    assert "runtime_graph_cache_drop_poisoned_entry" in captured.err, (
        "drop must emit a structured stderr event for observability; "
        f"saw stderr={captured.err!r}"
    )
