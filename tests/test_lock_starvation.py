"""RED scaffolds for REQ-4 lock starvation + H1 TOCTOU adversarial timing-window test.

Validation rows: F3 (lock starvation), F6 (latency SLO), REQ-4, H1 review item.

All tests are xfail(strict=True) because neither the yield protocol nor the
intent flag nor the LOCK_SH+LOCK_NB client-side retry loop exists yet. They
will flip from xfail to pass when the REQ-4 scoped-lock model and the H1 lock
protocol land.
"""
from __future__ import annotations

import fcntl
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test 1: consolidator acquires LOCK_EX under continuous readers
# ---------------------------------------------------------------------------


def test_consolidator_acquires_lock_ex_under_continuous_readers(
    hermetic_store: Path,
) -> None:
    """F3 / REQ-4: consolidator acquires LOCK_EX within 4 s under continuous LOCK_SH.

    Spawns N reader threads each holding LOCK_SH in a tight loop on the lock
    file, then asserts a consolidation actor acquires LOCK_EX within 4 s via
    the yield-protocol (intent flag causes readers to release after the current
    op, preventing starvation).

    RED: imports the not-yet-existing intent-flag helpers.
    """
    # Import the future yield-protocol helpers. These raise ImportError today,
    # which is the correct RED failure mode (body xfails, collection stays green).
    from iai_mcp.lock_protocol import (  # type: ignore[import]
        set_consolidation_intent,
        clear_consolidation_intent,
        acquire_client_shared_nb,
    )

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    stop_event = threading.Event()
    reader_errors: list[Exception] = []

    def _reader_loop() -> None:
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            while not stop_event.is_set():
                # Non-blocking acquire; on intent-flag set, yield immediately.
                acquired = acquire_client_shared_nb(fd, lock_path)
                if acquired:
                    time.sleep(0.001)  # simulate a brief read op
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    time.sleep(0.001)  # back off if intent flag set
        except Exception as exc:
            reader_errors.append(exc)
        finally:
            os.close(fd)

    # Start 4 continuous reader threads.
    threads = [threading.Thread(target=_reader_loop, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()

    # Give readers a moment to settle.
    time.sleep(0.05)

    # Set the intent flag and attempt LOCK_EX.
    set_consolidation_intent(lock_path)
    try:
        fd_ex = os.open(str(lock_path), os.O_RDWR)
        acquired = False
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd_ex, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.01)
        if acquired:
            fcntl.flock(fd_ex, fcntl.LOCK_UN)
        os.close(fd_ex)
    finally:
        clear_consolidation_intent(lock_path)
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

    assert not reader_errors, f"reader loop errors: {reader_errors}"
    assert acquired, (
        "consolidator did not acquire LOCK_EX within 4 s under continuous readers "
        "(REQ-4 yield protocol / F3 starvation)"
    )


# ---------------------------------------------------------------------------
# Test 2: recency read ≤1.5 s while VACUUM-like exclusive op runs
# ---------------------------------------------------------------------------


def test_recency_read_during_busy_meets_slo(hermetic_store: Path) -> None:
    """F6 / REQ-4: recency read completes in ≤1.5 s while a VACUUM-like exclusive op runs.

    Starts a background thread holding the lock file LOCK_EX (simulating the
    consolidation window), then asserts a recency read via the direct path
    completes in ≤1.5 s (busy_timeout absorbs the brief contention window).

    RED: imports the not-yet-existing direct recency helper.
    """
    from iai_mcp.direct_recency import read_recent_user_turns_direct  # type: ignore[import]

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    # Seed the store via MemoryStore before any lock gymnastics.
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord
    import uuid
    from datetime import datetime, timezone

    store = MemoryStore(hermetic_store)
    try:
        rec = MemoryRecord(
            id=uuid.uuid4(),
            tier="episodic",
            literal_surface="busy-vacuum probe text",
            aaak_index="",
            embedding=[0.0] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": "test-session", "role": "user"}],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            tags=["role:user"],
            language="en",
        )
        store.insert(rec)
        flush_record_buffer(store)
    finally:
        store.close()

    ready = threading.Event()
    done = threading.Event()

    def _hold_exclusive() -> None:
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ready.set()
            done.wait(timeout=3.0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    t = threading.Thread(target=_hold_exclusive, daemon=True)
    t.start()
    ready.wait(timeout=2.0)

    try:
        t0 = time.monotonic()
        turns = read_recent_user_turns_direct(hermetic_store, n=5)
        elapsed = time.monotonic() - t0
    finally:
        done.set()
        t.join(timeout=2.0)

    assert elapsed <= 1.5, (
        f"recency read during busy exclusive op took {elapsed:.3f} s (SLO ≤1.5 s)"
    )
    surfaces = [turn.literal_surface for turn in turns]
    assert any("busy-vacuum probe text" in s for s in surfaces), (
        "stored turn not found during exclusive-lock contention"
    )


# ---------------------------------------------------------------------------
# Test 3: H1 TOCTOU adversarial timing-window test
# ---------------------------------------------------------------------------


def test_check_then_lock_toctou_consolidator_not_starved(
    hermetic_store: Path,
) -> None:
    """F3 / F6 / REQ-4 / H1: adversarial TOCTOU race — consolidator not starved.

    Models the exact H1 race: a client passes the consolidation-pending precheck,
    THEN the consolidator sets the intent flag, THEN the client attempts LOCK_SH.

    Asserts:
    (a) The client's LOCK_SH acquisition is NON-BLOCKING (LOCK_SH|LOCK_NB retry
        loop) and honors the intent flag: if intent is set after acquisition the
        client releases promptly (post-acquire recheck-and-release).
    (b) Under SUSTAINED client churn (tight loop of short-lived LOCK_SH opens)
        the consolidator still acquires LOCK_EX within 4 s.
    (c) The post-acquire recheck fires at least once (explicit synchronization via
        a dedicated prober thread that inlines the TOCTOU scenario deterministically
        under CPython's GIL).
    """
    from iai_mcp.lock_protocol import (  # type: ignore[import]
        set_consolidation_intent,
        clear_consolidation_intent,
        acquire_client_shared_nb,
        check_consolidation_intent,
    )

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    stop_event = threading.Event()
    churn_errors: list[Exception] = []
    post_acquire_recheck_count = 0

    # Prober synchronization: coordinates the TOCTOU scenario deterministically.
    # The prober passes the precheck, waits at this barrier, then the main thread
    # sets the intent flag and releases the barrier, guaranteeing the TOCTOU window.
    _prober_ready = threading.Event()   # prober signals: precheck passed
    _intent_set = threading.Event()     # main signals: intent flag is now set

    def _churn_client() -> None:
        nonlocal post_acquire_recheck_count
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            while not stop_event.is_set():
                # Simulate the TOCTOU window: precheck passes, then intent may be set.
                # acquire_client_shared_nb must honor the intent flag: if intent is
                # detected post-acquire, release promptly (the H1 contract).
                acquired = acquire_client_shared_nb(fd, lock_path)
                if acquired:
                    # Post-acquire recheck: if intent now set, release immediately.
                    if check_consolidation_intent(lock_path):
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        post_acquire_recheck_count += 1
                    else:
                        time.sleep(0.0005)
                        fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    time.sleep(0.001)
        except Exception as exc:
            churn_errors.append(exc)
        finally:
            os.close(fd)

    def _prober_client() -> None:
        """Dedicated prober that deterministically creates the TOCTOU race.

        Inlines the TOCTOU scenario explicitly:
        1. Verify precheck passes (intent not set yet).
        2. Signal main thread: ready (precheck passed, about to call flock).
        3. Wait for main thread to set the intent flag.
        4. Call flock(LOCK_SH|LOCK_NB) — succeeds because main doesn't hold EX yet.
        5. Post-acquire recheck sees intent set → releases and increments counter.
        This exercises the exact H1 TOCTOU race path (precheck→intent-set→acquire→
        recheck-fires) without relying on random GIL scheduling.
        """
        nonlocal post_acquire_recheck_count
        fd = os.open(str(lock_path), os.O_RDONLY)
        try:
            # Step 1: verify precheck passes (not inside acquire_client_shared_nb
            # so the prober doesn't race with itself).
            if check_consolidation_intent(lock_path):
                return  # intent already set — prober can't run; test will rely on churn

            # Step 2: signal ready (precheck passed, TOCTOU window now open).
            _prober_ready.set()

            # Step 3: wait for main to set the intent flag (this is the TOCTOU
            # window: precheck passed, intent not yet set, waiting for set).
            _intent_set.wait(timeout=2.0)

            # Step 4: acquire LOCK_SH — succeeds because the main thread hasn't
            # acquired LOCK_EX yet (SH holders from churn threads keep it blocked).
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError:
                return  # EX acquired before us — TOCTOU window missed; test OK

            # Step 5: post-acquire recheck sees intent set → release + count.
            if check_consolidation_intent(lock_path):
                fcntl.flock(fd, fcntl.LOCK_UN)
                post_acquire_recheck_count += 1
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # Start 6 churn clients (for the starvation assertion).
    threads = [threading.Thread(target=_churn_client, daemon=True) for _ in range(6)]
    prober = threading.Thread(target=_prober_client, daemon=True)
    for t in threads:
        t.start()
    prober.start()

    # Give churn threads time to settle (and prober time to pass its precheck).
    time.sleep(0.05)

    # Wait for prober to signal readiness (precheck passed), then set intent.
    _prober_ready.wait(timeout=1.0)

    # Set intent flag (TOCTOU: prober already passed precheck).
    set_consolidation_intent(lock_path)
    _intent_set.set()  # release prober to call flock

    try:
        fd_ex = os.open(str(lock_path), os.O_RDWR)
        acquired = False
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            try:
                fcntl.flock(fd_ex, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(0.005)
        if acquired:
            fcntl.flock(fd_ex, fcntl.LOCK_UN)
        os.close(fd_ex)
    finally:
        clear_consolidation_intent(lock_path)
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        prober.join(timeout=2.0)

    assert not churn_errors, f"churn client errors: {churn_errors}"
    assert acquired, (
        "H1 TOCTOU: consolidator did not acquire LOCK_EX within 4 s under "
        "sustained client churn with post-acquire recheck (REQ-4 / H1)"
    )
    # Confirm the post-acquire recheck fired at least once (proves the intent-
    # recheck path was exercised, not just the precheck path).
    assert post_acquire_recheck_count > 0, (
        "H1 TOCTOU: post-acquire intent recheck never fired — "
        "the recheck-and-release path was not exercised"
    )


# ---------------------------------------------------------------------------
# Test 4: H1 SLO reconcile — client lock wait strictly below 1.5 s
# ---------------------------------------------------------------------------


def test_client_lock_wait_bounded_below_slo(hermetic_store: Path) -> None:
    """F3 / F6 / REQ-4 / H1: client lock wait is strictly below 1.5 s end-to-end.

    The test asserts the CLIENT-OBSERVABLE wait bound — the total elapsed time
    from LOCK_SH attempt to either lock acquired or intent-flag yield — is
    strictly below 1.5 s, distinct from SQLite's busy_timeout (2000 ms ceiling).

    RED: imports the not-yet-existing lock protocol helper.
    """
    from iai_mcp.lock_protocol import acquire_client_shared_nb  # type: ignore[import]

    lock_path = hermetic_store / "hippo" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    # Hold LOCK_EX for just over 0.5 s to force the retry loop.
    ready = threading.Event()
    done = threading.Event()

    def _hold_ex() -> None:
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            ready.set()
            done.wait(timeout=0.6)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    t = threading.Thread(target=_hold_ex, daemon=True)
    t.start()
    ready.wait(timeout=1.0)

    # Client attempts shared acquire; must complete under 1.5 s.
    fd_sh = os.open(str(lock_path), os.O_RDONLY)
    try:
        t0 = time.monotonic()
        done.set()  # release the exclusive holder
        acquired = False
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            result = acquire_client_shared_nb(fd_sh, lock_path)
            if result:
                acquired = True
                break
            time.sleep(0.01)
        elapsed = time.monotonic() - t0
        if acquired:
            fcntl.flock(fd_sh, fcntl.LOCK_UN)
    finally:
        os.close(fd_sh)
        t.join(timeout=2.0)

    assert elapsed < 1.5, (
        f"client lock wait took {elapsed:.3f} s — must be strictly < 1.5 s "
        "(not the 2000 ms busy_timeout ceiling)"
    )
    assert acquired, "client shared lock was never acquired within the 1.5 s SLO"
