"""bench/verbatim.py -- benchmark harness + diagnostics.

Simulates a session gap by inserting N pinned records, flooding the store with
`session_gap * noise_per_session` unrelated records, then retrieving each
pinned record by its own literal_surface as the cue. Counts byte-exact matches.

Target: >= ACCURACY_FLOOR (0.99) on pinned records --.

Exit codes:
- 0 if accuracy >= 0.99
- 1 otherwise

JSON output (one line to stdout):
    {"accuracy": float, "n_records": int, "session_gap": int,
     "hits_exact": int, "passed": bool, "floor": 0.99, "noise_mode": str,
     "skip_l0_seed": bool, "storage_direct": bool, "k": int}

 diagnostic flags -- BENCH-ONLY (no production change):
  --skip-l0-seed: skip _seed_l0_identity to isolate L0 crowding (effect b)
  --storage-direct: bypass recall(), call store.query_similar directly
                     (isolates provenance-write amplification, effect c)
  --n: override n_records (default 20)
  --gap: override session_gap (default 20)
  --noise-per-session: override noise_per_session (default 10)
  --k: override k_hits (default max(n_records + 10, 20))

Design note -- why we bypass dispatch("memory_recall"):
The Plan-02 core.memory_recall routes non-empty stores through recall_for_response
(entry-point split) which instantiates an Embedder() (downloads
bge-small-en-v1.5 from HuggingFace
on first call). That's fine for a real runtime but wrong for an offline bench:
we need to measure storage-layer verbatim-recall correctness, not embedder
warm-up latency. So we call `retrieve.recall` directly with a fixed cue
embedding aligned with the pinned records (all-ones vector).

H-03 noise model (review finding, 2026-04-16):
The original noise vector was [-0.5]^384, which gives cosine=-1.0 against the
[1.0]^384 cue -- making pinned-vs-noise discrimination a geometric artifact
rather than a measurement of the storage layer. The fix uses seeded
numpy.random.standard_normal(EMBED_DIM) normalised to unit length. Against a
[1.0]^384 cue the expected cosine of a random unit vector is 0 with stddev
1/sqrt(EMBED_DIM) ~= 0.05 -- realistic noise geometry, but pinned still wins
because cos=+1 >> cos~=0. The bench remains honest about what it measures
(literal_surface round-trip under realistic embedding noise, given a fixed
cue). A real bge-small-en-v1.5 bench is deferred to.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

# Resolve iai_mcp.* (via src) AND bench.* (via worktree root) to THIS
# worktree, not the parent venv's editable install. Idempotent: each
# `sys.path.insert` is guarded by an "if not already present" check.
import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from iai_mcp.core import _seed_l0_identity
from iai_mcp.retrieve import recall
from iai_mcp.store import EMBED_DIM, MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord

ACCURACY_FLOOR = 0.99   #
NOISE_SEED = 20260416   # fixed for reproducibility across runs / CI


def _make_pinned(text: str, dim: int = EMBED_DIM) -> MemoryRecord:
    """A pinned verbatim record -- detail_level=5, never_merge=True, never_decay=True.

    Uses a fixed all-ones embedding so the cue (also all-ones) maxes cosine to
    every pinned record simultaneously. The recall ranking then scores by
    insertion order / stability -- but the literal_surface substring match is
    the only correctness signal we care about.

    : language="en" required. `dim` parameterised so callers
    can match a legacy 384d store or the 1024d default; default is
    `EMBED_DIM` (the current module constant). Unit tests that construct a
    fresh isolated store pick up the default; bench main() queries the
    store instance's embed_dim so a pre-existing ~/.iai-mcp store (possibly
    still at 384d prior to migration) works unchanged.
    """
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] * dim,
        community_id=None,
        centrality=0.0,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["benchmark", "pinned"],
        language="en",
    )


def _random_unit_vector(rng: np.random.Generator, dim: int = EMBED_DIM) -> list[float]:
    """Unit-norm Gaussian vector with configurable dim.

    Expected cosine vs [1.0]^dim cue: 0 with stddev 1/sqrt(dim) ~= 0.05 at 384d
    or ~= 0.03 at 1024d. Uses the provided seeded Generator so every run
    reproduces identical noise.
    """
    v = rng.standard_normal(dim)
    v = v / np.linalg.norm(v)
    return v.tolist()


def _make_noise(i: int, rng: np.random.Generator, dim: int = EMBED_DIM) -> MemoryRecord:
    """Noise record with a random unit-vector embedding (H-03 honesty fix).

    Previous implementation used [-0.5]^EMBED_DIM which gave cosine=-1 against the
    cue, making pinned-vs-noise discrimination trivial by geometry. Seeded
    Gaussian unit vectors reproduce deterministically and approximate the
    orthogonality-on-average of real embeddings.

    : language="en" required.
    """
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"unrelated session noise record #{i}: " + ("y " * 20),
        aaak_index="",
        embedding=_random_unit_vector(rng, dim=dim),
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


def run_verbatim_bench(
    store: MemoryStore | None = None,
    n_records: int = 20,
    session_gap: int = 20,
    noise_per_session: int = 10,
    seed: int = NOISE_SEED,
    *,
    skip_l0_seed: bool = False,
    storage_direct: bool = False,
    k: int | None = None,
) -> dict:
    """Run the verbatim-recall benchmark.

    Parameters:
        store: optional; isolated tmp_path store in tests, default MemoryStore in CLI.
        n_records: how many pinned records to store and recall.
        session_gap: how many "sessions" of noise to interpose between write and recall.
        noise_per_session: noise records per simulated session.
        seed: RNG seed for noise vectors (H-03: reproducibility across runs).
        skip_l0_seed: effect (b) isolation -- skip the L0 identity
            seed so pinned records are not competed against by a fixed-embedding
            identity record. BENCH-SCOPE ONLY; production _seed_l0_identity is
            unchanged.
        storage_direct: effect (c) isolation -- bypass
            retrieve.recall() and call store.query_similar directly, so the
            per-hit provenance write amplification is removed from the hot loop.
            BENCH-SCOPE ONLY; production recall() is unchanged.
        k: override the top-k passed into recall(k_hits=K) or query_similar(k=K);
            None keeps the historic default of max(n_records + 10, 20).

    Returns a dict as documented in the module docstring.
    """
    s = store if store is not None else MemoryStore()
    if not skip_l0_seed:
        _seed_l0_identity(s)

    #: consult the store's actual embedding dim. An existing
    # store may still have 384d records pre--migration; a fresh store has
    # the default (1024d). Match either transparently.
    dim = s.embed_dim

    pinned_texts = [
        f"Alice said on day {i}: verbatim phrase #{i}-{'x' * 10}"
        for i in range(n_records)
    ]
    pinned_records = [_make_pinned(t, dim=dim) for t in pinned_texts]
    for r in pinned_records:
        s.insert(r)

    # Simulate session_gap * noise_per_session unrelated records.
    # H-03: seeded RNG shared across every noise draw so results are reproducible.
    rng = np.random.default_rng(seed)
    for session_idx in range(session_gap):
        for j in range(noise_per_session):
            s.insert(_make_noise(session_idx * noise_per_session + j, rng, dim=dim))

    # Post-Hippo: MemoryStore.insert is buffered — rows accumulate
    # in an in-process buffer that flushes on size (500) / time (5s) thresholds.
    # This bench inserts ~220 rows in well under a second, so neither threshold
    # fires before the recall loop below would read an empty hippo table.
    # Call flush explicitly here to restore the legacy "all inserts visible to
    # the next query" contract. Independent of any test-runner monkey-patch.
    flush_record_buffer(s)

    cue_emb = [1.0] * dim
    # k must be >= n_records for every pinned record to have a chance of surfacing.
    # Plus a buffer for the L0 seed + anti-hits tail, so we retrieve a generous top-k.
    effective_k = k if k is not None else max(n_records + 10, 20)
    hits_exact = 0
    for text in pinned_texts:
        if storage_direct:
            # (c): bypass recall() -> no per-hit provenance write amplification.
            raw = s.query_similar(cue_emb, k=effective_k)
            literal_surfaces = [rec.literal_surface for rec, _score in raw]
        else:
            #: retrieve.recall now defaults to mode='verbatim'
            # (conservative North-Star fallback). The bench's _make_pinned
            # uses tier='semantic' which the verbatim filter would drop.
            # The bench is measuring "verbatim TEXT exact-match recall under
            # noise" — that is independent of the cue-router's verbatim/concept
            # mode (the bench uses synthetic cues, not classifier-tagged
            # natural-language queries). Pin mode='concept' so the bench
            # measures what it has always measured.
            resp = recall(
                store=s,
                cue_embedding=cue_emb,
                cue_text=text,
                session_id="bench-verbatim",
                budget_tokens=5000,
                k_hits=effective_k,
                k_anti=3,
                mode="concept",
            )
            literal_surfaces = [h.literal_surface for h in resp.hits]
        if text in literal_surfaces:
            hits_exact += 1

    accuracy = hits_exact / n_records if n_records > 0 else 0.0
    return {
        "accuracy": accuracy,
        "n_records": n_records,
        "session_gap": session_gap,
        "noise_per_session": noise_per_session,
        "hits_exact": hits_exact,
        "passed": accuracy >= ACCURACY_FLOOR,
        "floor": ACCURACY_FLOOR,
        "noise_mode": "random-unit-vectors",
        "noise_seed": seed,
        # diagnostic traceability keys.
        "skip_l0_seed": bool(skip_l0_seed),
        "storage_direct": bool(storage_direct),
        "k": int(effective_k),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.verbatim",
        description="Verbatim recall benchmark with diagnostics",
    )
    parser.add_argument(
        "--skip-l0-seed",
        action="store_true",
        help="diagnostic: skip _seed_l0_identity to isolate L0 crowding effect",
    )
    parser.add_argument(
        "--storage-direct",
        action="store_true",
        help="diagnostic: bypass recall(), call store.query_similar directly",
    )
    parser.add_argument(
        "--n", "--n-records",
        dest="n_records",
        type=int,
        default=20,
        help="pinned record count (default 20)",
    )
    parser.add_argument(
        "--gap", "--session-gap",
        dest="session_gap",
        type=int,
        default=20,
        help="session gap -- how many noise sessions between writes and recall (default 20)",
    )
    parser.add_argument(
        "--noise-per-session",
        type=int,
        default=10,
        help="noise records per simulated session (default 10)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="override k_hits (default: max(n_records + 10, 20))",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = run_verbatim_bench(
        n_records=args.n_records,
        session_gap=args.session_gap,
        noise_per_session=args.noise_per_session,
        skip_l0_seed=args.skip_l0_seed,
        storage_direct=args.storage_direct,
        k=args.k,
    )
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
