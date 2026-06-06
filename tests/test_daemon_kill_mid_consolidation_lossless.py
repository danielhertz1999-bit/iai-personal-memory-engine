"""SIGKILL-mid-write losslessness gate for the store.

This is a SAFETY GATE. The watchdog is allowed to self-SIGKILL an
unresponsive (wedged) or memory-pressured store-owning process and let
the supervisor respawn it — but that auto-kill is ONLY legal if a kill
landing in the MIDDLE of a write/consolidation leaves the store lossless:
the SQLite source-of-truth and every verbatim record survive, the derived
ANN index is reconstructable at boot, and consolidation re-runs cleanly.

The constitutional claim being proven here:
  * SQLite (WAL, synchronous=NORMAL) is the durable source of truth.
  * The hnswlib ANN index is a DERIVED cache, rebuildable from SQLite
    BLOBs by the boot integrity rebuild on reopen.
  * Verbatim records are write-once and content-exact across a crash.

A FAILURE in this module is a HARD GATE, not a flake: if a SIGKILL mid-write
loses a record, mutates verbatim content, or leaves the index unrecoverable,
the auto-kill design is unsafe and the watchdog must NOT ship.

Falsifiability guards (so a green result cannot be a false green):
  * The child writes to the SAME on-disk store and only signals readiness
    AFTER its first committed insert, then keeps looping — so the kill lands
    while real writes are in flight, never in dead air.
  * After reopen we assert the active SQLite count GREW beyond the K pinned
    seeds; if it did not, the kill hit dead air and the run is a setup
    failure (not a pass).
  * The K seeds are pinned + never_merge + detail_level>=3 so consolidation
    cannot legitimately mutate them — any post-reopen content delta is
    unambiguous corruption, not normal lossy consolidation.

All hermetic: a tmp store, a spawned child PID, an explicit child env. The
SIGKILL targets the TEST child on the tmp store, never any real process or
real store. No remote subprocess (``claude -p``) can fire — the child does
not run the reconsolidation critic and the reopen pipeline run hard-stubs it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# Test passphrase MUST match the autouse conftest passphrase so the child
# process (a fresh interpreter that does NOT inherit the conftest
# monkeypatch'd defaults) derives the SAME AES-GCM key and its encrypted
# rows decrypt in the parent. Mismatch would surface as a confusing content
# mismatch rather than a lock/crash error.
_TEST_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30-phase-07.10"

K_SEEDS = 5
_CHILD_BARRIER_TIMEOUT_S = 60.0


def _seed_content(i: int) -> str:
    """A distinctive, per-record verbatim string used for content-exact checks."""
    return (
        f"alice pinned fact {i}: the lossless cat sat on durable mat number {i} "
        f"and the verbatim invariant held exactly token-{i}-{i * 7}"
    )


def _make_pinned_seed(i: int) -> MemoryRecord:
    """A pinned, never-merge, never-decay seed — consolidation may not mutate it."""
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=_seed_content(i),
        aaak_index="",
        # A deterministic unit-norm-ish vector; distinct per record so
        # query_similar can resolve each seed back out of the rebuilt index.
        embedding=[0.0] * i + [1.0] + [0.0] * (EMBED_DIM - i - 1),
        community_id=None,
        centrality=0.0,
        detail_level=3,  # >=3 forces never_decay True
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["seed", "lossless-gate"],
        language="en",
    )


# The child program: opens the SAME store EXCLUSIVE, commits a batch of churn
# records DURABLY to SQLite (an explicit flush — the in-process insert buffer
# is a within-process optimisation, NOT the durability boundary; the durable
# source of truth is SQLite, so we flush to model "records that reached the
# hippocampus"), signals readiness, then keeps writing in a tight loop until
# SIGKILL'd. Run as a fresh interpreter via subprocess so the kill is a real
# process kill (an in-process thread cannot be SIGKILL'd cleanly).
#
# CHURN seeds use orthogonal one-hot embeddings disjoint from the K pinned
# seeds' dimensions so pattern-separation never SKIP-merges them into a seed.
_CHILD_PROGRAM = r"""
import os, sys, time
from datetime import datetime, timezone
from uuid import uuid4

src = os.environ["IAI_MCP_TEST_SRC"]
if src not in sys.path:
    sys.path.insert(0, src)

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

store_root = os.environ["IAI_MCP_STORE"]
sentinel = os.environ["IAI_MCP_TEST_SENTINEL"]
n_durable = int(os.environ["IAI_MCP_TEST_CHURN_DURABLE"])
seed_dims = int(os.environ["IAI_MCP_TEST_SEED_DIMS"])  # dims reserved for K seeds


def _make(i):
    now = datetime.now(timezone.utc)
    # Place the one-hot at a dimension ABOVE the seeds' reserved range so churn
    # records are orthogonal to the K pinned seeds (no near-dup SKIP-merge).
    pos = seed_dims + (i % (EMBED_DIM - seed_dims))
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="alice churn write %d in flight" % i,
        aaak_index="",
        embedding=[0.0] * pos + [1.0] + [0.0] * (EMBED_DIM - pos - 1),
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
        tags=["churn"],
        language="en",
    )


store = MemoryStore(path=store_root)

# 1. Commit a batch of churn records DURABLY (explicit flush to SQLite) so the
#    store genuinely grew on disk before the kill — this is what survives.
for i in range(n_durable):
    store.insert(_make(i))
flush_record_buffer(store)

# 2. Signal readiness: durable rows are on disk; the parent may SIGKILL now.
with open(sentinel, "w") as fh:
    fh.write("ready")
    fh.flush()
    os.fsync(fh.fileno())

# 3. Keep writing forever (buffered + periodic flush) so the kill lands while
#    real index/write work is in flight.
i = n_durable
while True:
    store.insert(_make(i))
    i += 1
    if i % 20 == 0:
        flush_record_buffer(store)  # keep pushing durable rows under the kill
"""


# Number of churn records the child commits DURABLY (flushed to SQLite) before
# signalling readiness — proves the on-disk store grew beyond the K seeds.
CHURN_DURABLE = 40


def _spawn_writer_child(store_root: Path, sentinel: Path) -> subprocess.Popen:
    """Spawn a fresh-interpreter child that opens the store and churns writes."""
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_MCP_TEST_SENTINEL"] = str(sentinel)
    env["IAI_MCP_TEST_SRC"] = str(Path(__file__).resolve().parent.parent / "src")
    env["IAI_MCP_TEST_CHURN_DURABLE"] = str(CHURN_DURABLE)
    env["IAI_MCP_TEST_SEED_DIMS"] = str(K_SEEDS)
    # Match the parent's crypto key so the child's encrypted rows decrypt here.
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_PASSPHRASE
    # Belt-and-suspenders: the child never runs the reconsolidation critic, but
    # disable any remote tier defensively so no subscription call can fire.
    env["IAI_MCP_RECONSOLIDATION_TIER1"] = "0"
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD_PROGRAM],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _active_count(store: MemoryStore) -> int:
    row = store.db._conn.execute(
        "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
    ).fetchone()
    return int(row[0]) if row else 0


def test_sigkill_mid_write_leaves_store_lossless():
    """A SIGKILL while a writer is mid-write leaves the store lossless.

    Sequence:
      1. Parent seeds K pinned verbatim records, then closes (flush + unlock).
      2. A child process opens the store EXCLUSIVE and churns inserts, then
         signals readiness AFTER its first committed insert and keeps looping.
      3. Parent SIGKILLs the child while it is mid-write, then reaps it.
      4. Parent reopens the store fresh (the realistic SIGKILL path — the
         on-disk index is a valid-but-STALE file that loads without a rebuild)
         and asserts: opens clean; all K seeds present + content-EXACT (VERBATIM
         LOSSLESS AT BOOT — the no-data-loss claim); the active count GREW beyond
         K (the kill hit real durable work). The semantic ANN index is stale at
         boot (its raw count lags SQLite); it RECONCILES at the next
         consolidation (OPTIMIZE rebuilds hnswlib from SQLite) — after which the
         raw count equals the active count and a surviving CHURN record is
         resolvable via the index. Records are never lost; only the cache is
         briefly stale and self-heals on consolidation.
      5. Complementary hard proof that SQLite is the source of truth: delete the
         on-disk ANN index entirely, reopen, and assert the missing-file boot
         rebuild fires and reconstructs every record purely from SQLite BLOBs —
         the index is fully derived, the records are durable.

    A failure of any assertion is a HARD GATE: auto-kill is not lossless and
    must not ship. This is intentionally NOT marked xfail/skip.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="iai-lossless-gate-"))
    sentinel = tmp_root / ".child-ready"
    child: subprocess.Popen | None = None
    seed_ids: list[UUID] = []
    try:
        # 1. Seed K pinned verbatim records, then close to flush + release lock.
        seed_store = MemoryStore(path=tmp_root)
        for i in range(K_SEEDS):
            rec = _make_pinned_seed(i)
            seed_ids.append(rec.id)
            seed_store.insert(rec)
        seed_store.close()

        # 2. Spawn the churning writer child (opens EXCLUSIVE on the same store).
        child = _spawn_writer_child(tmp_root, sentinel)

        # Wait for the child's readiness sentinel (first insert committed).
        deadline = time.monotonic() + _CHILD_BARRIER_TIMEOUT_S
        while not sentinel.exists():
            if child.poll() is not None:
                out, err = child.communicate()
                raise AssertionError(
                    "writer child exited before signalling readiness "
                    f"(rc={child.returncode}); stderr=\n{err.decode(errors='replace')}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError("writer child never signalled readiness")
            time.sleep(0.01)

        # 3. SIGKILL the child while it is mid-write, then reap it so the kernel
        #    releases the flock before we reopen EXCLUSIVE.
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
        child = None

        # A churn-record probe vector: one-hot at the first churn dimension
        # (above the K seeds' reserved range), matching the child's churn
        # embedding placement, so a non-seed hit on this vector proves the
        # semantic index actually carries a child-written churn record.
        churn_vec = [0.0] * K_SEEDS + [1.0] + [0.0] * (EMBED_DIM - K_SEEDS - 1)

        # 4. Reopen fresh — the REALISTIC SIGKILL path: the on-disk index is a
        #    valid-but-STALE file (the last atomic save), so it loads without a
        #    rebuild. This is the durability boundary we actually care about.
        reopened = MemoryStore(path=tmp_root)
        try:
            # opens without raising == no corruption (reaching here proves it).

            # (a) VERBATIM LOSSLESS AT BOOT — the constitutional invariant. All K
            #     pinned seeds present AND content-EXACT (round-tripping the
            #     AES-GCM decrypt across the kill). This is the no-data-loss claim.
            for i, sid in enumerate(seed_ids):
                got = reopened.get(sid)
                assert got is not None, f"seed {i} lost after mid-write SIGKILL"
                assert got.literal_surface == _seed_content(i), (
                    f"seed {i} content corrupted after SIGKILL: "
                    f"{got.literal_surface!r}"
                )

            # (b) Falsifiability guard: the kill must have interrupted REAL work
            #     — the child committed durable churn rows beyond the K seeds to
            #     SQLite (the source of truth). If active == K, the kill hit dead
            #     air: a setup failure, NOT a pass.
            active = _active_count(reopened)
            assert active > K_SEEDS, (
                "SIGKILL hit dead air (no durable child write committed): "
                f"active={active} <= K={K_SEEDS}. Setup failure, NOT a pass."
            )

            # (c) The on-disk hnswlib index after a SIGKILL is STALE: it reflects
            #     the last atomic save, which may LAG the SQLite source of truth.
            #     The boot health "action" is count-based (label_map is
            #     repopulated from SQLite at open), so it reads "ok" even though
            #     the index raw count lags. We record this honestly: SQLite is
            #     the durable source of truth; the semantic index is a derived
            #     cache that self-heals at the NEXT consolidation (below), NOT at
            #     boot. The records are never lost — only the ANN cache is stale.
            from iai_mcp.daemon import _hippo_health_check_on_boot

            health = _hippo_health_check_on_boot(reopened)
            assert health["sqlite_count"] == active
            raw_at_boot = int(reopened.db._hnsw.get_current_count())

            # (d) THE REAL RECONCILIATION POINT: consolidation re-runs and its
            #     OPTIMIZE step rebuilds the hnswlib index from SQLite. The
            #     remote critic is hard-stubbed so no claude -p can fire.
            _run_consolidation_clean(reopened, tmp_root)

            # After consolidation, the index is reconciled to the full SQLite
            # source of truth: the raw index count now equals the active count
            # (it grew from the stale boot value), and a CHURN record (not just
            # a seed) is resolvable via the index — proving the semantic cache
            # was rebuilt to carry the child-written rows that survived the kill.
            raw_after = int(reopened.db._hnsw.get_current_count())
            assert raw_after == active, (
                f"index not reconciled by consolidation: raw={raw_after} != "
                f"active={active} (raw at boot was {raw_at_boot})"
            )
            assert raw_after >= raw_at_boot, "consolidation shrank the index"
            churn_hits = reopened.query_similar(churn_vec, n=5)
            assert any(r.id not in set(seed_ids) for r in churn_hits), (
                "no surviving churn record resolvable via the index after the "
                "post-kill consolidation rebuild"
            )
            # The seeds remain resolvable too (verbatim records are findable).
            for i, sid in enumerate(seed_ids):
                vec = [0.0] * i + [1.0] + [0.0] * (EMBED_DIM - i - 1)
                hit_ids = {r.id for r in reopened.query_similar(vec, n=3)}
                assert sid in hit_ids, (
                    f"seed {i} not resolvable after the consolidation rebuild"
                )
        finally:
            reopened.close()

        # 5. HARD proof SQLite is the source of truth: delete the derived ANN
        #    index file, reopen, and assert the boot rebuild reconstructs it
        #    from SQLite BLOBs and still resolves every seed.
        hnsw_path = tmp_root / "hippo" / "records.hnsw"
        if hnsw_path.exists():
            hnsw_path.unlink()
        rebuilt = MemoryStore(path=tmp_root)
        try:
            from iai_mcp.daemon import _hippo_health_check_on_boot

            health2 = _hippo_health_check_on_boot(rebuilt)
            assert health2["sqlite_count"] == health2["hnsw_active_count"], (
                f"index not rebuilt from SQLite after deletion: {health2}"
            )
            assert int(health2["sqlite_count"]) >= K_SEEDS
            for i, sid in enumerate(seed_ids):
                got = rebuilt.get(sid)
                assert got is not None, f"seed {i} lost after index-delete rebuild"
                assert got.literal_surface == _seed_content(i)
                vec = [0.0] * i + [1.0] + [0.0] * (EMBED_DIM - i - 1)
                hit_ids = {r.id for r in rebuilt.query_similar(vec, n=3)}
                assert sid in hit_ids, (
                    f"seed {i} not resolvable via index rebuilt purely from SQLite"
                )
        finally:
            rebuilt.close()
    finally:
        if child is not None:
            try:
                child.kill()
                child.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)


def _run_consolidation_clean(store: MemoryStore, tmp_root: Path) -> None:
    """Drive a full sleep-pipeline run on the reopened store; assert no error.

    Hard-stubs every remote subprocess entry point so no ``claude -p`` can
    fire, then runs the pipeline with tmp-redirected lifecycle/event paths.
    The stubs are scoped to this call via context managers so they never leak
    into other test modules collected in the same session.
    """
    import unittest.mock as mock

    import iai_mcp.claude_cli as _cc
    import iai_mcp.reconsolidation_critic as _rc
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.sleep_pipeline import SleepPipeline

    # Belt-and-suspenders remote-call stub: any subscription subprocess entry
    # point raises rather than spawning ``claude -p`` (mirrors the bench's
    # _install_remote_stubs). The critic gate is no-op'd so even a misconfigured
    # tier-1 cannot reach a subprocess.
    #
    # patch.object is used (not direct attribute assignment) so the originals are
    # automatically restored when the with-block exits, regardless of exceptions.
    def _raise_remote(*_a, **_k):  # pragma: no cover - must never be reached
        raise AssertionError("remote subprocess must not be invoked under test")

    log_dir = tmp_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_root / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=log_dir),
    )
    with (
        mock.patch.object(_cc, "invoke_claude_sync", _raise_remote),
        mock.patch.object(_cc, "invoke_claude_once", _raise_remote),
        mock.patch.object(_rc, "evaluate_batch_reconsolidation", lambda *_a, **_k: {}),
    ):
        result = pipeline.run()
    assert result.get("error") is None, (
        f"consolidation did not re-run cleanly after SIGKILL: {result.get('error')}"
    )
    assert int(result.get("critic_calls", 0) or 0) == 0, (
        "remote critic fired during the post-kill consolidation run"
    )
