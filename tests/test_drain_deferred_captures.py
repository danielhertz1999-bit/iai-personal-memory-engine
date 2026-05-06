"""Phase 7.1 Plan 06 / R3 closure — `drain_deferred_captures(store)` daemon-side.

Plan 07.1-05 shipped the WRITE side (`iai-mcp capture-transcript --no-spawn`
writes JSONL files to ``~/.iai-mcp/.deferred-captures/`` when the daemon
socket is unreachable). This plan ships the READ side: a drain function that
the daemon runs at startup AND on every WAKE-from-SLEEP transition, so
deferred events get ingested into the episodic tier within seconds of the
daemon coming back up.

End-to-end story this module verifies:
    user closes 3 sessions while daemon is sleeping
        → 3 Stop hooks fire `iai-mcp capture-transcript --no-spawn`
        → 3 JSONL deferral files appear under ~/.iai-mcp/.deferred-captures/
        → next MCP call socket-activates the daemon (or wake from idle)
        → drain runs → all 3 transcripts land in the brain
        → ZERO events lost; ZERO new daemons spawned

NOTE on idle-shutdown (per CONTEXT.md D7-05 inheritance): if the daemon
idle-exits cleanly while many hook deferrals accumulate, the deferred-
captures directory keeps growing until the NEXT non-hook MCP call
socket-activates the daemon. This is by design — eliminating the spawn
vector is the whole point. The drain happens whenever the daemon next runs.

Test layout:
    A: round-trip — write 3 events → drain → file deleted, store has records
    B: malformed event line — file renamed to .failed-<ts>, counts.files_failed=1
    C: forward-compat — version=99 header → file left in place + log entry
    D: missing dir — drain returns zero counts, no error
    E: empty file — drain unlinks it, counts unchanged
    F: multiple files — all 3 processed in glob-sort order, all deleted
    G: integration — daemon startup with malformed file pre-staged → daemon
       starts, malformed file is .failed-<ts>, daemon doesn't crash

Tests A–F are pure-Python unit tests of the drain function (in-process
MemoryStore, monkeypatch HOME/keyring). Test G is the integration check —
spawns a real `python -m iai_mcp.daemon` subprocess under env-isolation
(mirroring `test_doctor_apply_recovery.py:isolated_daemon_paths`) with a
malformed JSONL pre-seeded; asserts the daemon binds the socket without
crashing AND the malformed file is renamed to .failed-<ts>.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


REPO = Path(__file__).resolve().parent.parent

# POSIX-only: AF_UNIX socket + subprocess + Path-based glob semantics.
pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX socket; HOME isolation pattern",
)


# ---------------------------------------------------------------------------
# Fixture: HOME + keyring isolation for in-process tests (A–F)
# ---------------------------------------------------------------------------


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    """HOME=tmp_path + keyring fail-backend + crypto passphrase.

    The drain function uses ``Path.home()`` to find both
    ``.deferred-captures/`` and ``logs/`` — so HOME monkeypatching
    isolates from the user's real ~/.iai-mcp/.

    Drain calls ``capture_turn`` which calls ``store.insert()`` which
    encrypts via ``MemoryStore._key()`` → ``crypto.get_or_create()`` →
    keyring. Forcing the fail-backend + a passphrase env var sends us
    down the D-GUARD passphrase fallback so the macOS Security
    framework's interactive keychain prompt never fires.

    Returns ``tmp_path`` (also reachable via ``Path.home()`` thanks to
    monkeypatched ``HOME``).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-drain-passphrase")
    # IAI_MCP_STORE under tmp so a fresh LanceDB is created per test —
    # avoids cross-test row leakage.
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))

    # Force keyring to re-resolve the backend (it caches on first access).
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    # Reset post-test so the fail-backend cache doesn't leak.
    keyring.core._keyring_backend = None


# ---------------------------------------------------------------------------
# Helpers — JSONL fixture builders (D7.1-04 v1 format)
# ---------------------------------------------------------------------------


def _write_deferred_jsonl(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
    *,
    version: int = 1,
    ts_suffix: int | None = None,
) -> Path:
    """Construct a v1 JSONL file under ``deferred_dir`` and return its Path.

    Mirrors the format ``write_deferred_captures`` produces (Plan 07.1-05).
    Header on line 1; events on lines 2..N.
    """
    deferred_dir.mkdir(parents=True, exist_ok=True)
    suffix = ts_suffix if ts_suffix is not None else int(time.time())
    out = deferred_dir / f"{session_id}-{suffix}.jsonl"
    header = {
        "version": version,
        "deferred_at": "2026-04-26T00:00:00Z",
        "session_id": session_id,
        "cwd": "/tmp",
    }
    lines = [json.dumps(header)] + [json.dumps(e) for e in events]
    out.write_text("\n".join(lines) + "\n")
    return out


def _make_event(text: str, role: str = "user") -> dict:
    return {
        "text": text,
        "cue": f"test cue: {text[:24]}",
        "tier": "episodic",
        "role": role,
        "ts": "2026-04-26T00:00:00Z",
    }


def _open_isolated_store():
    """Construct a MemoryStore that respects the iai_home fixture's env.

    Imported lazily because module import touches LanceDB + crypto
    config; we want the env overrides in place first.
    """
    from iai_mcp.store import MemoryStore

    return MemoryStore()


# ---------------------------------------------------------------------------
# Test A — round-trip: write JSONL → drain → file deleted, store has records
# ---------------------------------------------------------------------------


def test_drain_consumes_jsonl_and_deletes_file(iai_home):
    """The happy path: drain reads a v1 JSONL, captures every event via
    capture_turn (so encryption + dedup + shield run), and unlinks the file.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    events = [
        _make_event("Alice said: drain test event one — must be at least 12 chars"),
        _make_event("assistant reply with sufficient length to pass MIN_CAPTURE", role="assistant"),
        _make_event("third event for the round-trip drain count assertion"),
    ]
    fpath = _write_deferred_jsonl(deferred_dir, "session-A", events)
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # W2 / counts schema split four ways per status.
    assert counts["files_drained"] == 1, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 3, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not fpath.exists(), "deferred file must be unlinked after drain"

    # Verify the events landed in the records table — count_rows is the
    # cheapest sanity check that drain actually inserted (capture_turn may
    # also reinforce/skip depending on dedup; for a fresh store all three
    # are net-new inserts).
    n_rows = store.db.open_table("records").count_rows()
    assert n_rows >= 3, f"expected ≥3 records inserted, got {n_rows}"


# ---------------------------------------------------------------------------
# Test B — malformed event line → file renamed to .failed-<ts>, count tallied
# ---------------------------------------------------------------------------


def test_drain_handles_malformed_event_line(iai_home):
    """Per-event JSON-decode failure surfaces as a per-FILE failure: drain
    catches the exception, renames the offender to .failed-<ts>, logs, and
    moves on. The original file MUST NOT exist after drain.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    # Hand-craft so we can inject a non-JSON line in the middle.
    fpath = deferred_dir / "session-B-12345.jsonl"
    fpath.write_text(
        json.dumps({
            "version": 1,
            "deferred_at": "2026-04-26T00:00:00Z",
            "session_id": "session-B",
            "cwd": "/tmp",
        }) + "\n"
        + json.dumps(_make_event("first valid event with adequate length")) + "\n"
        + "this line is not valid JSON {{{ broken\n"
        + json.dumps(_make_event("never reached because file-level error")) + "\n"
    )
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    assert counts["files_failed"] == 1, counts
    assert counts["files_drained"] == 0, counts
    # Original gone, .failed-<ts>.jsonl present (via with_suffix replacement).
    assert not fpath.exists(), "original must be renamed away on per-file error"
    failed = list(deferred_dir.glob("session-B-12345.failed-*.jsonl"))
    assert len(failed) == 1, f"expected exactly 1 .failed-* file, got {failed}"


# ---------------------------------------------------------------------------
# Test C — forward-compat: version > 1 → file left intact, log entry written
# ---------------------------------------------------------------------------


def test_drain_skips_future_version(iai_home):
    """A future-version header (version=99) is left in place so a newer
    daemon can handle it. Drain logs a "skip" line for forensic visibility.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-C",
        [_make_event("event from a future format version that we cannot parse")],
        version=99,
    )

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # W2 / counts schema split four ways per status.
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert fpath.exists(), "version>1 file must remain for a future daemon to handle"
    # No .failed-* either.
    assert not list(deferred_dir.glob("*.failed-*.jsonl"))

    # Log line should mention the file basename + version.
    log_dir = iai_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("deferred-drain-*.log"))
    assert log_files, "drain must create a log file when it skips a future version"
    log_content = log_files[0].read_text()
    assert "skip" in log_content
    assert "session-C" in log_content
    assert "version=99" in log_content


# ---------------------------------------------------------------------------
# Test D — no deferred dir → drain returns zero counts, no error
# ---------------------------------------------------------------------------


def test_drain_no_deferred_dir(iai_home):
    """Cold-boot path: ~/.iai-mcp/.deferred-captures/ doesn't exist yet.
    Drain must return zero counts cleanly without trying to mkdir or raise.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    assert not deferred_dir.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # W2 / counts schema split four ways per status.
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    # Drain MUST NOT auto-create the deferred dir — only the writer creates it.
    assert not deferred_dir.exists(), "drain should not create .deferred-captures/"


# ---------------------------------------------------------------------------
# Test E — empty (0-byte) file → drain unlinks it, counts unchanged
# ---------------------------------------------------------------------------


def test_drain_empty_jsonl(iai_home):
    """A 0-byte deferral file (e.g. from a writer that crashed before any
    line landed) is unlinked silently — no insert, no failure, no log.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    fpath = deferred_dir / "session-E-empty.jsonl"
    fpath.write_text("")  # 0 bytes
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # W2 / counts schema split four ways per status.
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 0, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert not fpath.exists(), "0-byte file must be unlinked"


# ---------------------------------------------------------------------------
# Test F — multiple files processed in glob-sort order, all deleted
# ---------------------------------------------------------------------------


def test_drain_multiple_files_processed_in_order(iai_home):
    """Three deferral files (sorted by name = sorted by unix_ts within a
    single session) are all drained in one pass. Counts aggregate correctly.
    """
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    # NOTE: 07.11-01 Rule 1 deviation -- before Plan 07.11-01 these three
    # lexically-near cues all looked unique because the dedup branch in
    # capture_turn was unreachable dead code (Bugs A/B/C). After the dedup
    # fix, bge-small-en-v1.5 places "test cue: event from file 0/1/2" above
    # the 0.95 cosine threshold and the second + third capture get correctly
    # de-duplicated -> events_inserted=1, events_reinforced=2.
    # The fix is to give each event a SEMANTICALLY divergent topic so cosine
    # genuinely separates them (matches the divergence pattern in
    # tests/test_capture_dedup_contract.py::test_capture_turn_inserts_on_low_cos).
    distinct_texts = [
        "apples are red and grow on trees in orchards across the world",
        "quantum chromodynamics describes the strong nuclear force precisely",
        "hummingbirds beat their wings about eighty times per second in flight",
    ]
    paths = []
    for i, base_ts in enumerate([1000, 2000, 3000]):
        events = [_make_event(distinct_texts[i])]
        paths.append(
            _write_deferred_jsonl(
                deferred_dir, f"session-F-{i}", events, ts_suffix=base_ts,
            )
        )
    assert all(p.exists() for p in paths)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # W2 / counts schema split four ways per status.
    assert counts["files_drained"] == 3, counts
    assert counts["events_inserted"] == 3, counts
    assert counts["events_skipped_insert_failed"] == 0, counts
    assert counts["files_failed"] == 0, counts
    for p in paths:
        assert not p.exists(), f"{p} must be unlinked after drain"


# ---------------------------------------------------------------------------
# Test H — W2 / per-event insert failure preserves the file
# ---------------------------------------------------------------------------


def test_drain_partial_insert_failure_preserves_file(iai_home, monkeypatch):
    """W2 / when ANY event in a file returns status=skipped reason=
    insert-failed:* (capture_turn swallowed a store.insert exception), the
    drain MUST rename the file to .failed-<ts>.jsonl and NOT unlink it.
    Pre-07.9 the file was deleted with the events permanently lost."""
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.store import MemoryStore

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"

    # File with three events: good, poison-sentinel (will fail insert), good.
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-H",
        [
            _make_event("first good event with adequate length here"),
            _make_event("INSERT_FAIL_SENTINEL_07_9 — this event triggers a failure"),
            _make_event("third good event after the failing one in the middle"),
        ],
        ts_suffix=42,
    )
    assert fpath.exists()

    # Patch MemoryStore.insert to raise when literal_surface contains the
    # sentinel string. This drives capture_turn into its insert-failed
    # return path (capture.py:169-171).
    real_insert = MemoryStore.insert

    def insert_or_fail(self, rec):
        if "INSERT_FAIL_SENTINEL_07_9" in rec.literal_surface:
            raise RuntimeError("simulated lance write failure")
        return real_insert(self, rec)

    monkeypatch.setattr(MemoryStore, "insert", insert_or_fail)

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # File NOT unlinked — renamed to .failed-<ts>.jsonl, evidence preserved.
    assert not fpath.exists(), "original file must be renamed when any insert fails"
    failed_files = list(deferred_dir.glob("session-H-42.failed-*.jsonl"))
    assert len(failed_files) == 1, (
        f"expected 1 .failed-* file; got {failed_files} "
        f"(deferred_dir contents: {list(deferred_dir.iterdir())})"
    )

    # Counts split four ways: 2 inserted (good ones), 1 insert-failed
    # (the sentinel), file marked failed (not drained).
    assert counts["events_inserted"] == 2, counts
    assert counts["events_skipped_insert_failed"] == 1, counts
    assert counts["events_skipped_intentional"] == 0, counts
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 1, counts

    # Log carries the insert-failed marker and the first error reason.
    log_dir = iai_home / ".iai-mcp" / "logs"
    log_files = list(log_dir.glob("deferred-drain-*.log"))
    assert log_files, "log file must record the insert-failed event"
    log_content = log_files[0].read_text()
    assert "insert-failed" in log_content
    assert "session-H" in log_content


# ---------------------------------------------------------------------------
# Test I — W2 / intentional skips do NOT fail the file
# ---------------------------------------------------------------------------


def test_drain_intentional_skip_does_not_fail_file(iai_home):
    """W2 / an event whose text is too short returns status=skipped
    reason='too short' — that's an INTENTIONAL skip, not an insert
    failure. The file must be unlinked normally; counts.files_failed=0;
    counts.events_skipped_intentional incremented."""
    from iai_mcp.capture import drain_deferred_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    fpath = _write_deferred_jsonl(
        deferred_dir,
        "session-I",
        [
            _make_event("ok this is a long enough event for the min-length gate"),
            # Too short event: will return status=skipped reason="too short".
            {"cue": "x", "text": "tiny", "tier": "episodic", "role": "user",
             "ts": "2026-04-26T00:00:00Z"},
        ],
        ts_suffix=43,
    )
    assert fpath.exists()

    store = _open_isolated_store()
    counts = drain_deferred_captures(store)

    # File unlinked: intentional skips DO NOT mark a file as failed.
    assert not fpath.exists()
    assert list(deferred_dir.glob("*.failed-*.jsonl")) == []
    assert counts["files_drained"] == 1, counts
    assert counts["files_failed"] == 0, counts
    assert counts["events_inserted"] == 1, counts
    assert counts["events_skipped_intentional"] == 1, counts
    assert counts["events_skipped_insert_failed"] == 0, counts


# ---------------------------------------------------------------------------
# Test G — integration: daemon startup with malformed file → daemon stays up,
# file is renamed to .failed-<ts>
# ---------------------------------------------------------------------------


# Mirror test_doctor_apply_recovery.py:isolated_daemon_paths so the spawned
# daemon writes its state + LanceDB + logs under tmp_path. Crucially this
# also propagates HF_HOME so the daemon's prewarm step (bge-small load)
# reuses the user's already-cached model and prewarm completes in <1s
# instead of trying to download from HuggingFace under an empty tmp HOME.


def _spawn_daemon(sock_path: Path, store_dir: Path, home: Path) -> subprocess.Popen:
    """Spawn `python -m iai_mcp.daemon` with full env-isolation."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    env["IAI_MCP_STORE"] = str(store_dir)
    env["IAI_DAEMON_IDLE_SHUTDOWN_SECS"] = "99999"
    # Reuse user's HF cache so bge-small doesn't redownload (pattern from
    # test_doctor_apply_recovery.py:69-89).
    env["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")
    # Force keyring fail-backend → passphrase fallback in the daemon
    # subprocess (otherwise macOS Security framework prompts interactively).
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-drain-integration-pass"
    return subprocess.Popen(
        [sys.executable, "-m", "iai_mcp.daemon"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_socket(sock_path: Path, timeout_sec: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if sock_path.exists():
            return True
        time.sleep(0.1)
    return False


def _kill_daemon_by_socket(sock_path: Path) -> None:
    """Match-by-env cleanup so we never touch the user's real daemon."""
    target = str(sock_path)
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if "iai_mcp.daemon" not in cl:
                continue
            try:
                env = p.environ()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if env.get("IAI_DAEMON_SOCKET_PATH") == target:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        p.send_signal(signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def test_daemon_main_drain_does_not_crash_on_bad_file(tmp_path, monkeypatch):
    """Pre-seed a malformed JSONL under .deferred-captures/ → spawn daemon.
    Daemon must (a) bind socket and stay alive, (b) rename the bad file to
    .failed-<ts>.jsonl. Confirms startup-drain's per-file try/except shields
    daemon main from a malformed input.
    """
    # Build the same env scaffolding as _spawn_daemon, applied to in-process
    # too so any pre-seed Path.home() lookups resolve to tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-drain-integration-pass")

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)
    store_dir = iai_dir / "lancedb"
    store_dir.mkdir(parents=True, exist_ok=True)
    deferred_dir = iai_dir / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed a malformed file BEFORE the daemon spawns.
    bad = deferred_dir / "session-G-99999.jsonl"
    bad.write_text(
        json.dumps({"version": 1, "session_id": "session-G",
                    "deferred_at": "2026-04-26T00:00:00Z", "cwd": "/tmp"}) + "\n"
        + "totally not JSON ===invalid===\n"
    )
    assert bad.exists()

    # Short socket path (macOS AF_UNIX 104-byte cap).
    sock_dir = Path(f"/tmp/iai-drn-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    proc = None
    try:
        proc = _spawn_daemon(
            sock_path, store_dir, home=Path(os.environ["HOME"])
        )
        assert _wait_for_socket(sock_path, timeout_sec=30), (
            f"daemon never bound socket within 30s; pid={proc.pid} "
            f"poll_status={proc.poll()}"
        )

        # Brief settle for startup-drain to run (asyncio.to_thread
        # immediately after daemon_started write_event).
        time.sleep(2.0)

        # Daemon process MUST still be alive (drain didn't crash it).
        assert proc.poll() is None, (
            f"daemon exited unexpectedly with code {proc.returncode} — "
            f"startup-drain probably propagated an exception"
        )

        # Bad file MUST be renamed to .failed-<ts>.jsonl.
        assert not bad.exists(), (
            "malformed file should have been renamed away by drain"
        )
        failed = list(deferred_dir.glob("session-G-99999.failed-*.jsonl"))
        assert len(failed) == 1, (
            f"expected exactly 1 .failed-* file, got {failed}"
        )
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait(timeout=3)
        _kill_daemon_by_socket(sock_path)
        try:
            if sock_path.exists():
                sock_path.unlink()
        except OSError:
            pass
        try:
            sock_dir.rmdir()
        except OSError:
            pass
        # Reset keyring cache.
        import keyring.core
        keyring.core._keyring_backend = None
