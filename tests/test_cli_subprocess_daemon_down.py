"""GENUINE subprocess CLI tests.

These invoke `python -m iai_mcp.iai_cli` as a REAL OS process (NOT cmd_*
function + monkeypatch).  The child process env sets ALL THREE hermetic guards:
  1. IAI_MCP_STORE → tmp store root
  2. IAI_DAEMON_SOCKET_PATH → a nonexistent path (socket call fails)
  3. HOME → tmp_home (so Path.home()/.iai-mcp/.deferred-captures resolves
     under tmp, NEVER the live ~/.iai-mcp)

Store-backed is asserted by CONTENT: a turn seeded ONLY in the tmp store
(drained — not in .live.jsonl, not in the bank) appears in the subprocess
stdout.  The live layer and bank cannot produce this text.

All tests are xfail(strict=True) until the corresponding CLI surface wiring lands.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Shared seeding + on-disk structural-cache helpers. The SAME definitions the
# in-process daemon-independent recall gate uses, so the real-subprocess gate
# below builds the IDENTICAL hub-sensitive gold layout + on-disk runtime-graph
# cache the recall path consumes. Built ON DISK (not a monkeypatch), so a real
# subprocess that honors no test stub still finds the cache.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _recall_helpers import (  # noqa: E402
    UUID_TWO_HOP_SURFACE,
    _populate_store,
    _prime_structural_cache,
)

# Same passphrase conftest's autouse _crypto_passphrase_env installs for the
# parent test process — the child subprocess opens the SAME tmp store, so it
# must decrypt with the SAME key. (conftest's monkeypatch does not cross a
# subprocess boundary, so the child env sets it explicitly.)
_TEST_CRYPTO_PASSPHRASE = "iai-mcp-test-passphrase-2026-04-30-phase-07.10"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _child_env(store_root: Path, tmp_home: Path) -> dict[str, str]:
    """Build a hermetic child-process env.

    Copies the parent env (so iai_mcp is importable), then overrides:
    - HOME → tmp_home (Path.home()/.iai-mcp resolves under tmp)
    - IAI_MCP_STORE → store_root
    - IAI_DAEMON_SOCKET_PATH → a nonexistent path (socket fails immediately)
    """
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(tmp_home / "no-such-daemon.sock")
    return env


def _seed_store_with_drained_turn(store_root: Path, text: str) -> None:
    """Insert a turn directly into the tmp store (simulating a drained turn).

    A drained turn is in the SQLite store but NOT in .live.jsonl and NOT in
    bank — the live-layer fallback in cmd_last and bank-recall cannot see it.
    """
    import numpy as np
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(store_root)
    try:
        rng = np.random.RandomState(seed=88)
        vec = rng.randn(EMBED_DIM).tolist()
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=vec,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "c3h1-session", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        # Release LOCK_EX before spawning the child process.
        store.close()


# ---------------------------------------------------------------------------
# Test 1: `iai last` subprocess — daemon down — store-backed drained turn
# ---------------------------------------------------------------------------


def test_subprocess_iai_last_daemon_down_returns_drained_store_turn(
    hermetic_store: Path, tmp_path: Path
) -> None:
    """`python -m iai_mcp.iai_cli last` with daemon down returns drained store turn.

    Seeds a distinctive turn ONLY in the tmp store (drained).
    Runs `python -m iai_mcp.iai_cli last` as a real subprocess with the daemon
    socket forced absent and HOME overridden.
    Asserts stdout CONTAINS the distinctive turn text (store-backed by content).
    """
    tmp_home = tmp_path / "tmp_home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    drained_text = "c3h1 last drained distinctive store turn text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    env = _child_env(hermetic_store, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "last", "--n", "10"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"subprocess `iai last` failed (rc={result.returncode}):\n{result.stderr}"
    )
    assert drained_text in result.stdout, (
        f"drained store turn not in `iai last` stdout;\n"
        f"stdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}\n"
        "The live-layer fallback cannot produce this turn — must be store-backed."
    )


# ---------------------------------------------------------------------------
# Test 2: `iai capture` subprocess — daemon down — turn written to store
# ---------------------------------------------------------------------------


def test_subprocess_iai_capture_daemon_down_writes_to_store(
    hermetic_store: Path, tmp_path: Path
) -> None:
    """`python -m iai_mcp.iai_cli capture` with daemon down writes to store.

    Runs `python -m iai_mcp.iai_cli capture "<text>"` as a real subprocess with
    the daemon down.  Asserts:
    (1) subprocess exit code is 0 (no hard-fail);
    (2) the captured row is present in the tmp Hippo store.

    Proves store-backed by opening the tmp store in-test after the subprocess
    completes and reading the row.
    """
    tmp_home = tmp_path / "tmp_home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    capture_text = "c3h1 capture distinctive write probe text"
    env = _child_env(hermetic_store, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "capture", capture_text],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"subprocess `iai capture` failed (rc={result.returncode}):\n{result.stderr}\n"
        "capture must succeed (exit 0) even with daemon down; "
        "the direct-write fallback is not yet wired."
    )

    # Open the tmp store in-test and verify the row exists.
    from iai_mcp.store import MemoryStore

    store = MemoryStore(hermetic_store)
    try:
        records = store.all_records()
        surfaces = [r.literal_surface or "" for r in records]
        assert any(capture_text in s for s in surfaces), (
            f"captured text not found in tmp Hippo store after subprocess capture;\n"
            f"surfaces={surfaces!r}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3: `iai recall` subprocess — daemon down — store-backed degraded result
# ---------------------------------------------------------------------------


def test_subprocess_iai_recall_daemon_down_returns_store_backed_degraded(
    hermetic_store: Path, tmp_path: Path
) -> None:
    """`python -m iai_mcp.iai_cli recall` with daemon down returns store-backed result.

    Seeds a distinctive turn ONLY in the tmp store (drained; not in bank).
    Runs `python -m iai_mcp.iai_cli recall "<cue>"` as a real subprocess with
    the daemon down.  Asserts stdout CONTAINS the distinctive turn's text
    (store-backed degraded by content — bank-recall cannot produce it).

    Hermeticity: HOME=tmp_home in the child env so
    Path.home()/.iai-mcp/.deferred-captures resolves under tmp.  No
    deferred-event fixtures are needed here (no live events expected).
    """
    tmp_home = tmp_path / "tmp_home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    drained_text = "c3h1 recall store backed degraded distinctive probe text"
    _seed_store_with_drained_turn(hermetic_store, drained_text)

    env = _child_env(hermetic_store, tmp_home)

    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "recall", "c3h1 recall store backed"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"subprocess `iai recall` failed (rc={result.returncode}):\n{result.stderr}"
    )
    assert drained_text in result.stdout, (
        f"drained store turn not in `iai recall` stdout;\n"
        f"stdout={result.stdout!r}\n"
        f"stderr={result.stderr!r}\n"
        "The bank-recall subprocess cannot produce this turn — must be store-backed."
    )


# ---------------------------------------------------------------------------
# Scripted LIVE gate: real `iai recall --json` subprocess, daemon DOWN,
# pre-built on-disk structural cache, real offline embedder construct ->
# EXACT _source == "daemon-down-full" + the STRUCTURAL-ONLY 2-hop gold present.
#
# This is the deterministic equivalent of the cold `iai recall` daemon-asleep
# LIVE check for the DOWN/HIBERNATED case: the daemon process has exited
# (hibernation) -> socket dead -> the memory_recall RPC fails the well-formed
# check -> clean fall-through to the daemon-independent construct path. We
# reproduce it hermetically (tmp store + dead socket) but with a REAL embedder
# construct and the REAL structural pipeline — proving the symptom is gone
# end-to-end, not in a monkeypatched in-process harness.
# ---------------------------------------------------------------------------


def _hf_cache_root() -> Path:
    """Resolve the on-disk HF weight cache (portable — never a hardcoded path).

    The Rust embedder loads bge-small via the hf_hub crate, whose cache the
    loader resolves from ``HF_HOME`` (falling back to ``~/.cache/huggingface``
    when unset). The child env sets ``HF_HOME`` to this root, so the cached
    weights are reachable under a hermetic tmp HOME without any symlink. This
    helper returns the real cache root.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface"


def _live_gate_child_env(store_root: Path, tmp_home: Path) -> dict[str, str]:
    """Hermetic child env for the daemon-down-full LIVE gate.

    Hermetic guards (tmp): HOME, IAI_MCP_STORE, IAI_DAEMON_SOCKET_PATH (dead).
    The ONLY parent-system crossover is the read-only HF weight cache, reached
    via ``HF_HOME`` (the Rust loader honors ``HF_HOME``, so the cached weights
    are reachable under a hermetic tmp HOME with no symlink; ``HF_HUB_CACHE`` /
    ``HUGGINGFACE_HUB_CACHE`` are set too for forward-compat with newer hf-hub —
    the pinned 0.4.3 reads only ``HF_HOME``). IAI_MCP_EMBED_OFFLINE=1 makes the
    construct fully offline + deterministic
    (no flaky network ETAG roundtrip). IAI_MCP_AROUSAL_USE_SHADOW=1 is the
    sanctioned test isolation so the 2-hop structural spread is not gated by an
    arousal cosine rank_threshold (the structural-only gold sits at cosine
    ~0.02).
    """
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    env["IAI_MCP_STORE"] = str(store_root)
    env["IAI_DAEMON_SOCKET_PATH"] = str(tmp_home / "no-such-daemon.sock")
    env["IAI_MCP_EMBED_OFFLINE"] = "1"
    env["IAI_MCP_AROUSAL_USE_SHADOW"] = "1"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = _TEST_CRYPTO_PASSPHRASE
    # Forward-compat only (no-op against the current home-relative Rust loader);
    # harmless once the loader becomes env-aware (Cache::from_env hardening).
    hf_root = _hf_cache_root()
    env["HF_HOME"] = str(hf_root)
    env["HF_HUB_CACHE"] = str(hf_root / "hub")
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")
    return env


def test_subprocess_iai_recall_daemon_down_returns_daemon_down_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LIVE gate: real `iai recall --json` subprocess, daemon DOWN, returns EXACT
    daemon-down-full + the STRUCTURAL-ONLY 2-hop gold present.

    The verification-integrity gate (the most scrutinized fix). The load-bearing
    assertion is the STRUCTURAL-ONLY UUID(5) gold PRESENT — reachable ONLY via
    the 2-hop / rich-club spread, NOT the ANN top-K — which is the causal,
    telemetry-independent proof the real subprocess fed the construct into the
    FULL structural pipeline. The exact _source == "daemon-down-full" is
    SUPPORTING evidence (the source string is force-stamped on any non-empty
    hit, so it can NOT be the sole assertion). An ANN-only fall-through
    (_source == "direct-store") would MISS UUID(5); a recency degrade
    (_source == "daemon-down-degrade") would not run the structural pipeline at
    all; a daemon answer (_source == "daemon") is impossible here (dead socket).

    The real subprocess honors no test monkeypatch: it constructs a REAL
    Embedder() offline from the symlinked weight cache (~67 ms warm) and runs the
    real recall path. The store + on-disk structural cache are seeded in THIS
    (parent) process with the SAME real-embedder cue vector so real cosine hits
    survive rank_threshold and the seeded gold geometry matches what the
    subprocess's real embedder reproduces for the same cue string.
    """
    # The cached bge-small weights must be present on disk; otherwise the offline
    # construct would fail -> degrade, and this gate would test nothing. Skip
    # honestly (a weight-less machine), do NOT silently degrade to a recency pass.
    hf_cache = _hf_cache_root()
    weights_dir = hf_cache / "hub" / "models--BAAI--bge-small-en-v1.5"
    if not weights_dir.exists():
        pytest.skip(
            f"bge-small weight cache absent ({weights_dir}); the offline LIVE-gate "
            "construct cannot run. The authoritative real-hibernated-daemon proof "
            "is the human-live checkpoint (orchestrator), which this gate approximates."
        )

    store_root = tmp_path / "store"
    tmp_home = tmp_path / "home"
    tmp_home.mkdir(parents=True, exist_ok=True)

    # The hermetic tmp HOME reaches the real weights via HF_HOME (set in the
    # child env below): the Rust loader honors HF_HOME / HF_HUB_CACHE, so no
    # HOME-relative symlink is needed. Read-only crossover; no store/daemon
    # contact.

    # --- PARENT-PROCESS SEED: real-embedder cue vector + on-disk structural cache.
    # Seed the gold geometry collinear with the REAL Embedder().embed(cue) vector
    # so the subprocess's real embedder (deterministic, same weights) reproduces
    # the same cue vector and the seeded ANN/2-hop geometry holds. n_filler=700
    # places the structural-only UUID(5) (cosine ~0.02) OUTSIDE the ANN top-K, so
    # the 2-hop spread is genuinely load-bearing. The parent seeds offline too
    # (HF_HOME honored, IAI_MCP_EMBED_OFFLINE=1) so the seed is deterministic with
    # no network roundtrip and matches the child's offline construct exactly.
    # FORCE the passphrase the child env hardcodes (_live_gate_child_env) so the
    # parent (writer) and child (reader) derive the IDENTICAL AES key. A bare
    # ``os.environ.setdefault`` here is a no-op when a predecessor module planted
    # a DIFFERENT passphrase via its own module-import ``os.environ.setdefault``
    # (these are never reverted). The parent then encrypted the store + rgc cache
    # under the leaked passphrase while the child decrypts under this one →
    # ``cryptography.exceptions.InvalidTag`` in the child → the structural pipeline
    # crashes and the recall silently degrades (2-hop gold missing, full-suite
    # only). ``monkeypatch.setenv`` forces the correct value AND auto-reverts on
    # teardown, leak-immune both ways.
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", _TEST_CRYPTO_PASSPHRASE)
    monkeypatch.setenv("HF_HOME", str(hf_cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hf_cache / "hub"))
    monkeypatch.setenv("IAI_MCP_EMBED_OFFLINE", "1")
    from iai_mcp.embed import Embedder
    from iai_mcp.store import MemoryStore

    cue = "User reference gold document semantic recall probe cue"
    cue_vec = Embedder().embed(cue)

    from uuid import UUID

    from iai_mcp.pipeline import K_CANDIDATES

    store = MemoryStore(str(store_root))
    try:
        _populate_store(store, cue_vec=cue_vec, n_filler=700)
        _prime_structural_cache(store)

        # PRECONDITION (gives the gate teeth for THIS real-embedder cue_vec): the
        # structural-only UUID(5) gold must NOT be a direct ANN top-K hit, so its
        # presence in the result below can ONLY be the 2-hop / rich-club spread —
        # not the ANN pool. (Holds by construction: UUID(5)·cue == 0.02 for any
        # cue_vec; fillers are cue-independent, so the ~0.03 top-K cutoff excludes
        # 0.02. Asserted explicitly so the proof is self-evident, not implicit.)
        ann_top_k = {r.id for r, _ in store.query_similar(cue_vec, k=K_CANDIDATES)}
        assert UUID(int=5) not in ann_top_k, (
            f"PRECONDITION FAILED: the structural-only gold UUID(5) is a DIRECT ANN "
            f"top-{K_CANDIDATES} hit — the 2-hop spread would not be load-bearing and "
            f"the gate would be hollow. store size={store.active_records_count()}."
        )
    finally:
        # Release LOCK_EX before spawning the child (mirrors
        # _seed_store_with_drained_turn above) — else the subprocess blocks/fails
        # on the store lock.
        store.close()

    env = _live_gate_child_env(store_root, tmp_home)

    # --limit 50 so the structural-only UUID(5) is inside the returned window:
    # its degree-boost score (~0.12) clears the bulk of fillers but sits below the
    # high-cosine filler tail and the direct/intermediate gold — landing outside
    # the default top-5 but within top-50. It enters the result pool ONLY via the
    # 2-hop spread (cosine 0.02, outside ANN top-K). Mirrors the in-process gate's
    # n=50. NOT a latency window.
    result = subprocess.run(
        [sys.executable, "-m", "iai_mcp.iai_cli", "recall", "--json", "--limit", "50", cue],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"subprocess `iai recall --json` failed (rc={result.returncode}):\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )

    # The JSON payload is the LAST stdout line (degraded-path breadcrumbs go to
    # stderr, but parse defensively against any stray stdout noise).
    stdout_lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert stdout_lines, f"no JSON on stdout; stderr={result.stderr!r}"
    payload = json.loads(stdout_lines[-1])

    source = payload.get("_source")
    hits = payload.get("hits") or []
    surfaces = {h.get("literal_surface", "") for h in hits}

    # --- LOAD-BEARING: the STRUCTURAL-ONLY 2-hop gold is PRESENT (construct-engaged
    # proof — an ANN-only fall-through would MISS it). This, not the source string,
    # certifies the construct fed the FULL structural pipeline.
    assert UUID_TWO_HOP_SURFACE in surfaces, (
        "VERIFICATION-INTEGRITY FAILURE: the STRUCTURAL-ONLY 2-hop gold "
        f"({UUID_TWO_HOP_SURFACE!r}) is MISSING from the real-subprocess "
        "daemon-down recall. It is reachable ONLY via the 2-hop / rich-club "
        "spread (cosine ~0.02, outside ANN top-K), so its absence means the "
        "construct did NOT feed the full structural pipeline — the "
        f"daemon-down-full label would be hollow.\n_source={source!r}\n"
        f"gold surfaces present={sorted(s for s in surfaces if 'gold doc' in s)}\n"
        f"stderr={result.stderr!r}"
    )

    # --- EXACT supporting source label (no loose negation).
    assert source == "daemon-down-full", (
        f"expected EXACT _source == 'daemon-down-full', got {source!r}.\n"
        f"stderr={result.stderr!r}"
    )

    # --- The source is NOT any path that would mean the construct/structural
    # pipeline did not engage (documents the guards; the structural-gold assertion
    # above is the causal proof, these are corroborating).
    assert source != "daemon", "a daemon answered — impossible with a dead socket"
    assert source != "direct-store", "ANN-only fall-through (structural pipeline skipped)"
    assert source != "daemon-down-degrade", "recency degrade (no structural pipeline ran)"

    # --- Non-empty + non-zero top score.
    assert int(payload.get("count", 0)) > 0, f"empty recall; payload={payload!r}"
    top_score = hits[0].get("score") if hits else None
    assert top_score is not None and float(top_score) != 0.0, (
        f"top hit must have a non-zero score; got {top_score!r}"
    )
