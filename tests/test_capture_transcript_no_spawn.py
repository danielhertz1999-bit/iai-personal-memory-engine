"""Acceptance tests for `iai-mcp capture-transcript --no-spawn`.

Eliminates the third spawn vector: the Stop-hook
spawning iai_mcp.daemon under N-session race. When 3 Claude sessions close
within seconds, 3 hooks each fire `iai-mcp capture-transcript --no-spawn`;
ZERO daemons get spawned. Each invocation either (a) talks to the existing
daemon if one is up, or (b) writes a JSONL deferral file and exits 0 within
2s. The hook never blocks session teardown.

This module covers:
  - Test A: writes deferred file when daemon is unreachable
  - Test B: completes in under 2s wall-clock (budget)
  - Test C: spawns ZERO new iai_mcp.* processes
  - Test D: --no-spawn surfaces in --help; default (no flag) keeps
            behavior (exit 0 + stdout JSON, no deferred file)
  - Test E: deferred JSONL v1 header + per-turn event lines
  - Test F: missing transcript -> header-only file, no exception

Test isolation:
  - HOME=tmp_path so `Path.home()` resolves to a fresh dir; the user's
    real ~/.iai-mcp/.deferred-captures/ is never touched.
  - IAI_DAEMON_SOCKET_PATH=/tmp/iai-no-spawn-<pid>-<n>/d.sock so the
    250ms socket probe never hits the user's real daemon.
  - Subprocess invocation: `[sys.executable, '-m', 'iai_mcp.cli',...]`
    with PYTHONPATH set; we don't depend on the `iai-mcp` console script
    being on PATH (test_socket_subagent_reuse.py pattern).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent

# POSIX-only: subprocess + AF_UNIX socket probe; fork-style daemon counts.
pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX subprocess + AF_UNIX",
)


# ---------------------------------------------------------------------------
# Helpers (copied from test_socket_subagent_reuse.py to keep this module
# standalone — that test owns the canonical pattern, but cross-importing
# would couple two unrelated test modules).
# ---------------------------------------------------------------------------


def _count_iai_mcp_processes() -> dict[str, int]:
    """Snapshot iai_mcp.core / iai_mcp.daemon process counts on host."""
    counts = {"core": 0, "daemon": 0}
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info.get("cmdline") or []
            if not cl:
                continue
            joined = " ".join(c or "" for c in cl)
            if "iai_mcp.core" in joined:
                counts["core"] += 1
            if "iai_mcp.daemon" in joined:
                counts["daemon"] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return counts


def _isolated_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Build env that isolates HOME + socket path to tmp_path. Returns
    (env_dict, deferred_dir). Forces the keyring fail-backend so any
    accidental MemoryStore() doesn't prompt the macOS keychain.
    """
    sock_dir = Path(f"/tmp/iai-no-spawn-{os.getpid()}-{id(tmp_path)}")
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock_path = sock_dir / "d.sock"

    iai_dir = tmp_path / ".iai-mcp"
    iai_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["IAI_DAEMON_SOCKET_PATH"] = str(sock_path)
    # Defense-in-depth: if the inline path is somehow exercised, force the
    # fail-backend so we don't hang on the real keychain prompt.
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    env["IAI_MCP_CRYPTO_PASSPHRASE"] = "test-no-spawn-pass"
    # Make the spawned python find iai_mcp without an editable install.
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")

    return env, iai_dir / ".deferred-captures"


def _make_transcript(tmp_path: Path) -> Path:
    """Write a 3-turn Claude Code-style JSONL transcript."""
    turns = [
        {"type": "user", "message": {"role": "user", "content": "hello world"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "hi back at you"}},
        {"type": "user", "message": {"role": "user", "content": "third turn here"}},
    ]
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text("\n".join(json.dumps(t) for t in turns) + "\n")
    return transcript_path


def _run_no_spawn(env: dict[str, str], transcript_path: Path) -> subprocess.CompletedProcess:
    """Invoke `iai-mcp capture-transcript --no-spawn <transcript>` via
    `python -m iai_mcp.cli`. 5s wall-clock budget — well above the 2s
    contract the implementation must meet.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "capture-transcript",
            "--no-spawn",
            "--session-id",
            "test-r3",
            str(transcript_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )


# ---------------------------------------------------------------------------
# Subprocess tests (Tests A-D).
# ---------------------------------------------------------------------------


def test_no_spawn_writes_deferred_when_daemon_down(tmp_path):
    """Test A: --no-spawn writes a JSONL deferral file when daemon unreachable."""
    env, deferred_dir = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    proc = _run_no_spawn(env, transcript)

    assert proc.returncode == 0, f"stderr={proc.stderr!r} stdout={proc.stdout!r}"
    payload = json.loads(proc.stdout.strip())
    assert payload.get("status") == "deferred", payload

    files = sorted(deferred_dir.glob("*.jsonl"))
    assert len(files) == 1, f"expected 1 deferral file, got {files}"

    out_path = files[0]
    lines = out_path.read_text().splitlines()
    assert len(lines) >= 2, f"expected header + ≥1 event, got {lines}"

    header = json.loads(lines[0])
    assert header["version"] == 1, header
    assert header["session_id"] == "test-r3", header
    assert "deferred_at" in header
    assert "cwd" in header

    # Subsequent lines are events with text/cue/tier/role/ts.
    for line in lines[1:]:
        ev = json.loads(line)
        assert "text" in ev and ev["text"], ev
        assert ev["tier"] == "episodic", ev
        assert ev["role"] in {"user", "assistant"}, ev


def test_no_spawn_completes_in_under_2s(tmp_path):
    """Test B: wall-clock under 2s."""
    env, _ = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    t0 = time.time()
    proc = _run_no_spawn(env, transcript)
    duration = time.time() - t0

    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    assert duration < 2.0, (
        f"--no-spawn took {duration:.3f}s; budget is <2.0s. "
        f"Hook would block session teardown."
    )


def test_no_spawn_does_not_spawn_daemon(tmp_path):
    """Test C: ZERO new iai_mcp.* processes appear after invocation."""
    env, _ = _isolated_env(tmp_path)
    transcript = _make_transcript(tmp_path)

    before = _count_iai_mcp_processes()
    proc = _run_no_spawn(env, transcript)
    # Brief settle for any would-be spawn; cap at 0.5s — if a daemon were
    # going to appear, it would be visible within this window (psutil enum
    # picks up forked children immediately).
    time.sleep(0.5)
    after = _count_iai_mcp_processes()

    assert proc.returncode == 0, f"stderr={proc.stderr!r}"

    # Delta-snapshot: assert no new daemon or core processes appeared.
    delta_daemon = after["daemon"] - before["daemon"]
    delta_core = after["core"] - before["core"]
    assert delta_daemon <= 0, (
        f"--no-spawn spawned {delta_daemon} new daemon(s); spawn budget violated. "
        f"before={before} after={after}"
    )
    assert delta_core <= 0, (
        f"--no-spawn spawned {delta_core} new core(s); spawn budget violated. "
        f"before={before} after={after}"
    )


def test_no_spawn_flag_default_false(tmp_path):
    """Test D: --no-spawn appears in --help; default path keeps behavior.

    Per design, capture_transcript() returns a JSON dict with errors=1
    on missing transcript and the CLI prints that to stdout (NOT stderr).
    Default invocation without --no-spawn must:
      - exit 0 (fail-safe hook contract)
      - produce JSON-parsable stdout
      - NOT create any deferred-captures file (only --no-spawn does that)
    """
    env, deferred_dir = _isolated_env(tmp_path)

    # 1) --help advertises --no-spawn.
    help_proc = subprocess.run(
        [sys.executable, "-m", "iai_mcp.cli", "capture-transcript", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert help_proc.returncode == 0, help_proc.stderr
    assert "--no-spawn" in help_proc.stdout, help_proc.stdout

    # 2) Default path with non-existent transcript: behavior.
    default_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.cli",
            "capture-transcript",
            str(tmp_path / "no-such-file.jsonl"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert default_proc.returncode == 0, default_proc.stderr

    # prints the {errors: N,...} JSON to STDOUT, not stderr.
    # We just need it to be valid JSON with no.deferred-captures created.
    payload = json.loads(default_proc.stdout.strip())
    assert "errors" in payload or "inserted" in payload, payload

    # CRITICAL: default path must NOT write a deferred-captures file.
    if deferred_dir.exists():
        assert not list(deferred_dir.glob("*.jsonl")), (
            f"default capture-transcript must not write deferred files; got "
            f"{list(deferred_dir.glob('*.jsonl'))}"
        )


# ---------------------------------------------------------------------------
# Pure unit tests of write_deferred_captures (Tests E and F).
# ---------------------------------------------------------------------------


def test_deferred_jsonl_format_v1_header(tmp_path, monkeypatch):
    """Test E: write_deferred_captures emits v1 header + 1 event per turn."""
    monkeypatch.setenv("HOME", str(tmp_path))

    transcript = _make_transcript(tmp_path)

    from iai_mcp.capture import write_deferred_captures

    out_path = write_deferred_captures(
        session_id="unit-e",
        transcript_path=transcript,
        cwd="/some/cwd",
    )

    assert out_path.exists()
    assert out_path.parent == tmp_path / ".iai-mcp" / ".deferred-captures"
    # Filename pattern: <session_id>-<unix_ts>.jsonl
    assert out_path.name.startswith("unit-e-"), out_path.name
    assert out_path.suffix == ".jsonl", out_path.name

    lines = out_path.read_text().splitlines()
    # Header + 3 events (one per turn from _make_transcript).
    assert len(lines) == 4, lines

    header = json.loads(lines[0])
    assert header["version"] == 1
    assert header["session_id"] == "unit-e"
    assert header["cwd"] == "/some/cwd"
    assert "deferred_at" in header

    # Subsequent lines carry the event schema.
    for ln in lines[1:]:
        ev = json.loads(ln)
        assert set(ev.keys()) >= {"text", "cue", "tier", "role", "ts"}, ev.keys()
        assert ev["tier"] == "episodic"
        assert ev["role"] in {"user", "assistant"}
        assert ev["text"] in {"hello world", "hi back at you", "third turn here"}


def test_deferred_jsonl_handles_missing_transcript(tmp_path, monkeypatch):
    """Test F: missing transcript -> header-only file, no exception, exit 0 path."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from iai_mcp.capture import write_deferred_captures

    # Should NOT raise; should return a Path; file should exist with header only.
    out_path = write_deferred_captures(
        session_id="unit-f",
        transcript_path=tmp_path / "does-not-exist.jsonl",
    )

    assert out_path.exists()
    lines = out_path.read_text().splitlines()
    assert len(lines) == 1, f"expected header-only, got {lines}"

    header = json.loads(lines[0])
    assert header["version"] == 1
    assert header["session_id"] == "unit-f"
    # cwd defaults to os.getcwd() when not passed — non-empty string.
    assert isinstance(header.get("cwd"), str) and header["cwd"], header
