from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import numpy as np

from test_store import _make  # noqa: E402

EMBED_DIM = 384
RNG_SEED = 20260601

UUID_HUB = UUID(int=2)
UUID_SEED = UUID(int=3)
UUID_INTER = UUID(int=4)
UUID_TWO_HOP = UUID(int=5)

UUID_TWO_HOP_SURFACE = "User reference gold doc 5"


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _deterministic_vec(seed: int = 12345) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_gold_record(i: int, vec: list[float]):
    from iai_mcp.types import MemoryRecord
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=f"User reference gold doc {i}",
        aaak_index="",
        embedding=vec,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _populate_store(store, cue_vec: list[float], n_filler: int = 300) -> None:
    from iai_mcp.types import MemoryRecord as _MR

    rng = np.random.default_rng(RNG_SEED)
    cue_arr = np.asarray(cue_vec, dtype=np.float32)
    cue_arr /= np.linalg.norm(cue_arr)

    for i in range(n_filler):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        store.insert(_make(text=f"User filler record {i}", vec=v.tolist()))

    store.insert(_make_gold_record(1, cue_arr.tolist()))

    hub_rng = np.random.default_rng(44444)
    hub_vec = hub_rng.random(EMBED_DIM).astype(np.float32)
    hub_vec /= np.linalg.norm(hub_vec)
    store.insert(_make_gold_record(2, hub_vec.tolist()))
    store.boost_edges([(UUID(int=2), UUID(int=1))], edge_type="hebbian", delta=[3.0])

    seed3_rec = _MR(
        id=UUID(int=3), tier="episodic",
        literal_surface="User reference gold doc 3",
        aaak_index="", embedding=cue_arr.tolist(), community_id=None,
        centrality=0.0, detail_level=2, pinned=False, stability=0.0,
        difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
        provenance=[], created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), tags=[], language="en",
    )
    store.insert(seed3_rec)

    inter_rng = np.random.default_rng(55555)
    inter_noise = inter_rng.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_arr) * cue_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_vec = 0.4 * cue_arr + 0.9165 * inter_noise
    inter_vec /= np.linalg.norm(inter_vec)
    store.insert(_make_gold_record(4, inter_vec.tolist()))

    two_hop_rng = np.random.default_rng(66666)
    noise = two_hop_rng.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_arr) * cue_arr
    noise /= np.linalg.norm(noise)
    target_cos = 0.02
    orth_mag = float(np.sqrt(max(0.0, 1.0 - target_cos**2)))
    two_hop_vec = target_cos * cue_arr + orth_mag * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))

    store.boost_edges([(UUID(int=3), UUID(int=4))], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(UUID(int=4), UUID(int=5))], edge_type="hebbian", delta=[5.0])

    rng_boost = np.random.default_rng(77001)
    for boost_i in range(50):
        raw_v = rng_boost.random(EMBED_DIM).astype(np.float32)
        raw_v -= np.dot(raw_v, cue_arr) * cue_arr
        raw_v /= np.linalg.norm(raw_v)
        boost_rec = _MR(
            id=UUID(int=100 + boost_i), tier="episodic",
            literal_surface=f"User boost helper {boost_i}",
            aaak_index="", embedding=raw_v.tolist(), community_id=None,
            centrality=0.0, detail_level=2, pinned=False, stability=0.0,
            difficulty=0.0, last_reviewed=None, never_decay=False, never_merge=True,
            provenance=[], created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc), tags=[], language="en",
        )
        store.insert(boost_rec)
        store.boost_edges(
            [(UUID(int=5), UUID(int=100 + boost_i))], edge_type="hebbian", delta=[2.0]
        )


def _prime_structural_cache(store) -> None:
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, assignment, rc)
