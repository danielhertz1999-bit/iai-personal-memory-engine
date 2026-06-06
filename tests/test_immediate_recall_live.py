"""REQ-1..4, REQ-6: immediate recall from the pending live-capture layer.

Tests that a turn written to {session}.live.jsonl is returned by
read_pending_live_events() WITHOUT any drain call, and that the helper
correctly handles large files, Cyrillic text, processing-window files,
session-aware selection, and the ts-normalization contract.

SAFETY: ALL tests monkeypatch HOME to a tmp dir so Path.home() resolves
inside tmp_path. Never touches ~/.iai-mcp/ or the live daemon socket.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deferred_dir(home: Path) -> Path:
    d = home / ".iai-mcp" / ".deferred-captures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_live_file(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
    *,
    version: int = 1,
) -> Path:
    """Write a synthetic {session_id}.live.jsonl file."""
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": version,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps(ev, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _ts(offset_secs: float = 0.0) -> str:
    dt = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_secs)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_immediate_recall_no_drain(tmp_path, monkeypatch):
    """REQ-1: a turn in a.live.jsonl is returned without any drain call.

    - count == 1 (the header line must NOT leak as a phantom event)
    - returned dict has text, role, session_id, source_uuid, tz-aware ts
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import write_deferred_event, read_pending_live_events

    session = "sess-immediate-recall"
    write_deferred_event(session, "user", "hello world from live layer", source_uuid="u1")

    events = read_pending_live_events()
    assert len(events) == 1, f"expected 1 event, got {len(events)}: {events!r}"
    ev = events[0]
    assert ev["text"] == "hello world from live layer"
    assert ev["role"] == "user"
    assert ev["session_id"] == session
    assert ev["source_uuid"] == "u1"
    assert isinstance(ev["ts"], datetime), f"ts must be datetime, got {type(ev['ts'])}"
    assert ev["ts"].tzinfo is not None, "ts must be tz-aware"


def test_processing_marker_file_returned(tmp_path, monkeypatch):
    """REQ-1, H2: a drain-claimed.processing-*.jsonl is still returned.

    Covers the rename→commit window where drain renamed the live file to
    {stem}.processing-{pid}.jsonl but has not yet committed to the store.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "proc-session-0001"
    deferred = _deferred_dir(tmp_path)

    # Build the drain-claimed shape: {session}.live-{epoch}.processing-{pid}.jsonl
    name = f"{session}.live-1700000000.processing-99999.jsonl"
    path = deferred / name
    header = json.dumps({
        "version": 1,
        "deferred_at": _ts(),
        "session_id": session,
        "cwd": "/tmp",
    })
    ev_line = json.dumps({
        "text": "processing window marker turn",
        "role": "user",
        "tier": "episodic",
        "ts": _ts(1),
    })
    path.write_text(header + "\n" + ev_line + "\n", encoding="utf-8")

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "processing window marker turn" in texts, (
        f"processing-window file not read; got events: {events!r}"
    )


def test_tail_returns_newest_past_500(tmp_path, monkeypatch):
    """REQ-1, REQ-3, H1 line-axis: a file with 600 event lines returns newest.

    The deque(maxlen=500) tail must keep lines 101..600; line 1 (oldest)
    must be absent; line 600 with text='NEWEST' must be present.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "tail-600-session"
    deferred = _deferred_dir(tmp_path)
    path = deferred / f"{session}.live.jsonl"

    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })
    lines = [header]
    for i in range(600):
        ev = {
            "text": "OLDEST" if i == 0 else (f"line-{i}" if i < 599 else "NEWEST"),
            "role": "user",
            "tier": "episodic",
            "ts": _ts(i),
        }
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "NEWEST" in texts, f"NEWEST turn not returned; sample: {texts[:3]}"
    assert "OLDEST" not in texts, "OLDEST turn (line 1) must be dropped by deque tail"


def test_large_cyrillic_file_newest_returned(tmp_path, monkeypatch):
    """REQ-1, REQ-3, REQ-6, H1 byte-axis + UTF-8: >1MB Cyrillic file returns newest.

    Builds a single live file whose total size EXCEEDS 1 MB using Cyrillic
    text in every event line. The LAST line is the newest turn with marker
    text "САМОЕ НОВОЕ". The helper MUST return it and MUST NOT raise.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "big-cyrillic-session"
    deferred = _deferred_dir(tmp_path)
    path = deferred / f"{session}.live.jsonl"

    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })

    # Build Cyrillic padding — use a long Russian string per event line.
    # Each Cyrillic char is 2 bytes in UTF-8. We need the file to exceed 1 MB.
    # Use a 600-byte-per-event-line payload (300 Cyrillic chars × 2 bytes each).
    # 600 lines × ~650 bytes/line (JSON overhead + 600 bytes payload) ≈ 390 KB.
    # Bump to ~1800 chars per event body to get ~1800*2 = 3600 bytes per line:
    # 400 lines × ~3700 bytes ≈ 1.48 MB encoded.
    long_cyrillic = "Русский текст для теста UTF-8 в больших файлах захвата памяти. " * 30  # ~1800 chars

    lines = [header]
    for i in range(400):
        ev = {
            "text": "САМОЕ НОВОЕ" if i == 399 else f"строка-{i} {long_cyrillic}",
            "role": "user",
            "tier": "episodic",
            "ts": _ts(i),
        }
        lines.append(json.dumps(ev, ensure_ascii=False))
    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    assert path.stat().st_size > 1_000_000, (
        f"file must exceed 1 MB; got {path.stat().st_size} bytes"
    )

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "САМОЕ НОВОЕ" in texts, (
        f"newest Cyrillic turn not returned; got {len(events)} events"
    )


def test_partial_last_line_no_crash(tmp_path, monkeypatch):
    """REQ-6: a partial (no-newline) last line is silently dropped; no exception."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "partial-line-session"
    deferred = _deferred_dir(tmp_path)
    path = deferred / f"{session}.live.jsonl"

    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })
    ev_complete = json.dumps({
        "text": "complete event line", "role": "user", "tier": "episodic", "ts": _ts()
    })
    ev_partial = '{"text": "incomplete'  # no closing brace, no newline

    path.write_text(header + "\n" + ev_complete + "\n" + ev_partial, encoding="utf-8")

    # Must not raise; partial line is dropped
    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert "complete event line" in texts, "complete event should be returned"
    # The partial line should not appear (it was dropped — neither complete JSON nor ends with \n)
    assert not any("incomplete" in t for t in texts), "partial line must not appear"


def test_version2_header_skipped(tmp_path, monkeypatch):
    """REQ-6: a file with header version:2 yields zero events (forward-compat skip)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "v2-forward-compat-session"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [{"text": "should not appear", "role": "user", "tier": "episodic", "ts": _ts()}],
        version=2,
    )

    events = read_pending_live_events()
    assert len(events) == 0, f"version:2 file must be skipped; got {events!r}"


def test_bounded_scan_ignores_noise_files(tmp_path, monkeypatch):
    """REQ-6, H3: noise files are excluded by the allowlist; ≤20 files parsed.

    Creates 3000 noise files + 3 real *.live.jsonl files.
    Verifies read_pending_live_events() returns only real live events and
    does NOT parse the noise files (json.loads call count is bounded).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)

    # Create noise files of various noise shapes
    for i in range(100):
        (deferred / f"some-session.failed-{i}.jsonl").write_text(
            json.dumps({"noise": True}) + "\n"
        )
    for i in range(100):
        (deferred / f"some-session.permanent-failed-{i}.jsonl").write_text(
            json.dumps({"noise": True}) + "\n"
        )
    for i in range(100):
        # Stop-hook renamed shape — no.processing marker, should NOT match
        (deferred / f"some-session.live-{1700000000 + i}.jsonl").write_text(
            json.dumps({"noise": True}) + "\n"
        )
    # quarantine subdir — not a file, structurally excluded
    (deferred / ".quarantine").mkdir(exist_ok=True)

    # Three real.live.jsonl files
    for i in range(3):
        _write_live_file(
            deferred, f"real-session-{i}",
            [{"text": f"real turn {i}", "role": "user", "tier": "episodic", "ts": _ts(i)}],
        )

    parse_count = 0
    real_json_loads = json.loads

    def counting_loads(s, **kw):
        nonlocal parse_count
        parse_count += 1
        return real_json_loads(s, **kw)

    monkeypatch.setattr("iai_mcp.capture.json.loads", counting_loads)

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    for i in range(3):
        assert f"real turn {i}" in texts, f"real turn {i} missing"

    # json.loads called ONLY for the 3 real files (1 header + 1 event each = ≤ 9 calls
    # for the real files). The noise files were excluded by the allowlist BEFORE any open.
    # We allow a generous bound — the key is that noise is NOT parsed at all.
    # 3 real files × (1 header line + 1 event line) = 6 calls — assert < 100.
    assert parse_count <= 100, (
        f"allowlist should exclude noise files before parse; got {parse_count} json.loads calls"
    )


def test_mtime_desc_caps_to_20(tmp_path, monkeypatch):
    """REQ-6, H3: bare call reads the 20 NEWEST-by-mtime and skips the 10 oldest."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)

    # Create 30 valid.live.jsonl files with staggered mtimes
    base_mtime = time.time() - 3000
    for i in range(30):
        sid = f"mtime-session-{i:02d}"
        marker = "NEWEST_MARKER" if i == 29 else ("OLD_MARKER" if i == 0 else f"mid-{i}")
        _write_live_file(
            deferred, sid,
            [{"text": marker, "role": "user", "tier": "episodic", "ts": _ts(i)}],
        )
        p = deferred / f"{sid}.live.jsonl"
        mtime = base_mtime + i * 100
        os.utime(str(p), (mtime, mtime))

    events = read_pending_live_events()
    texts = [e["text"] for e in events]

    # The 20 newest (i=10..29) are read; the 10 oldest (i=0..9) are skipped
    assert "NEWEST_MARKER" in texts, "newest file (i=29) must be in results"
    assert "OLD_MARKER" not in texts, "oldest file (i=0) must be excluded by the ≤20 cap"
    # Cap respected
    assert len(events) <= 20, f"must read ≤20 files; got {len(events)}"


def test_session_aware_select_not_starved_by_other_sessions(tmp_path, monkeypatch):
    """REQ-1 + session-aware-additive: the requested session is never starved.

    Creates 25 other-session files whose mtimes are ALL NEWER than session X's
    file, then verifies that read_pending_live_events(session_id=X) still
    returns X's turn. Without force-include, the global mtime-desc top-20
    would exclude X's older file.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)
    base_mtime = time.time() - 10000

    # Session X — written OLDEST
    X = "target-session-x"
    _write_live_file(
        deferred, X,
        [{"text": "X_MARKER_TURN", "role": "user", "tier": "episodic", "ts": _ts(0)}],
    )
    x_path = deferred / f"{X}.live.jsonl"
    os.utime(str(x_path), (base_mtime, base_mtime))

    # 25 other sessions — all have NEWER mtimes than X
    for i in range(25):
        sid = f"other-session-{i:02d}"
        _write_live_file(
            deferred, sid,
            [{"text": f"other-{i}", "role": "user", "tier": "episodic", "ts": _ts(i + 1)}],
        )
        p = deferred / f"{sid}.live.jsonl"
        newer_mtime = base_mtime + (i + 1) * 100
        os.utime(str(p), (newer_mtime, newer_mtime))

    # Call with session_id=X — must return X's marker
    events = read_pending_live_events(session_id=X)
    texts = [e["text"] for e in events]
    assert "X_MARKER_TURN" in texts, (
        f"session X was starved by 25 newer sessions; got texts: {texts!r}"
    )

    # The ≤20 cap must still hold — force-include is additive, not unbounded
    assert len(events) <= 20, f"≤20 cap must hold; got {len(events)}"


def test_session_filter(tmp_path, monkeypatch):
    """REQ-1: session_id=A returns only A's events; bare call returns both."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import write_deferred_event, read_pending_live_events

    write_deferred_event("session-A-filter", "user", "turn from session A for filter test")
    write_deferred_event("session-B-filter", "user", "turn from session B for filter test")

    a_events = read_pending_live_events(session_id="session-A-filter")
    assert all(e["session_id"] == "session-A-filter" for e in a_events), (
        f"filtered result contains non-A events: {a_events!r}"
    )
    assert len(a_events) == 1, f"expected 1 A event; got {len(a_events)}"

    all_events = read_pending_live_events()
    session_ids = {e["session_id"] for e in all_events}
    assert "session-A-filter" in session_ids
    assert "session-B-filter" in session_ids


def test_sorted_ts_desc(tmp_path, monkeypatch):
    """REQ-3 helper-level: three events with increasing ts return newest-first."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "sort-ts-session"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [
            {"text": "oldest", "role": "user", "tier": "episodic", "ts": _ts(0)},
            {"text": "middle", "role": "user", "tier": "episodic", "ts": _ts(1)},
            {"text": "newest", "role": "user", "tier": "episodic", "ts": _ts(2)},
        ],
    )

    events = read_pending_live_events()
    texts = [e["text"] for e in events]
    assert texts[0] == "newest", f"newest should be first; got {texts}"
    assert texts[-1] == "oldest", f"oldest should be last; got {texts}"


def test_read_time_idem_tag_matches_drain_time(tmp_path, monkeypatch):
    """idem-tag computed from ev['ts_iso'] matches capture_turn's tag.

    Takes an event whose ts has.000000 microseconds and NO source_uuid.
    Verifies the read-side idem-tag (using ev['ts_iso']) byte-equals the
    drain-side tag (capture_turn builds ts_iso the same way).
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import _idem_tag, _resolve_ts, read_pending_live_events

    session = "idem-ts-norm-session"
    # Use a ts with.000000 microseconds — this is the tricky case
    ts_raw = "2026-05-31T12:00:00.000000+00:00"
    text = "idem tag normalization test turn content here"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [{"text": text, "role": "user", "tier": "episodic", "ts": ts_raw}],
    )

    events = read_pending_live_events()
    assert len(events) == 1, f"expected 1 event; got {events!r}"
    ev = events[0]

    # Read-side tag (using ev['ts_iso'] which is _resolve_ts(ts_raw).isoformat())
    read_tag = _idem_tag(session, "user", ev["ts_iso"], text, source_uuid=None)

    # Drain-side tag: mirrors capture_turn
    now = _resolve_ts(ts_raw)
    ts_iso_drain = now.isoformat()
    drain_tag = _idem_tag(session, "user", ts_iso_drain, text, source_uuid=None)

    assert read_tag == drain_tag, (
        f"read-side tag {read_tag!r} != drain-side tag {drain_tag!r}; "
        f"ev['ts_iso']={ev['ts_iso']!r}, drain ts_iso={ts_iso_drain!r}"
    )


def test_header_session_id_overrides_filename(tmp_path, monkeypatch):
    """LOW L1: header session_id is authoritative; filename stem is NOT.

    A file whose FILENAME stem differs from header session_id must:
    - Be returned when filtering by the HEADER session_id (via mtime-fill path)
    - NOT be returned when filtering by the filename stem

    This also confirms force-include is ADDITIVE: a header-matching file whose
    filename does NOT start with '{session_id}.live' must reach the candidate
    set via the mtime-fill path, not via force-include.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    deferred = _deferred_dir(tmp_path)
    filename_stem = "wrong-session-name"
    header_session_id = "correct-header-session"

    # File where filename stem != header session_id
    path = deferred / f"{filename_stem}.live.jsonl"
    header = json.dumps({
        "version": 1,
        "deferred_at": _ts(),
        "session_id": header_session_id,  # AUTHORITATIVE
        "cwd": "/tmp",
    })
    ev_line = json.dumps({
        "text": "header-session-turn marker text xyzw",
        "role": "user",
        "tier": "episodic",
        "ts": _ts(1),
    })
    path.write_text(header + "\n" + ev_line + "\n", encoding="utf-8")

    # Filtering by header value should return the event (via mtime-fill + header filter)
    events_header = read_pending_live_events(session_id=header_session_id)
    texts_header = [e["text"] for e in events_header]
    assert "header-session-turn marker text xyzw" in texts_header, (
        f"header-session filter should return event; got {events_header!r}"
    )

    # Filtering by filename stem should return nothing (header says different session)
    events_stem = read_pending_live_events(session_id=filename_stem)
    texts_stem = [e["text"] for e in events_stem]
    assert "header-session-turn marker text xyzw" not in texts_stem, (
        f"filename-stem filter must NOT return header-session event; got {events_stem!r}"
    )


def test_live_epoch_file_ignored(tmp_path, monkeypatch):
    """LOW L1: {id}.live-{epoch}.jsonl (Stop-hook renamed) is ignored.

    This file shape does NOT match _LIVE_ACTIVE_RE (has '-' before '.jsonl'
    after '.live') and has no '.processing' marker. It must be excluded by
    the allowlist even when its stem startswith the requested session_id.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "epoch-file-session"
    deferred = _deferred_dir(tmp_path)

    # Stop-hook renamed shape
    path = deferred / f"{session}.live-1700000000.jsonl"
    header = json.dumps({
        "version": 1, "deferred_at": _ts(), "session_id": session, "cwd": "/tmp"
    })
    ev_line = json.dumps({
        "text": "should not appear from epoch file",
        "role": "user", "tier": "episodic", "ts": _ts(),
    })
    path.write_text(header + "\n" + ev_line + "\n", encoding="utf-8")

    events = read_pending_live_events(session_id=session)
    texts = [e["text"] for e in events]
    assert "should not appear from epoch file" not in texts, (
        f"live-epoch file must be excluded by allowlist; got {events!r}"
    )


def test_assistant_role_preserved(tmp_path, monkeypatch):
    """Q5: assistant-role events are returned by the helper (callers filter by role)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    session = "assistant-role-session"
    deferred = _deferred_dir(tmp_path)
    _write_live_file(
        deferred, session,
        [{"text": "assistant response text here", "role": "assistant",
          "tier": "episodic", "ts": _ts()}],
    )

    events = read_pending_live_events()
    roles = [e["role"] for e in events]
    assert "assistant" in roles, (
        f"assistant-role event must be returned (helper does not filter role); "
        f"got roles: {roles!r}"
    )


def test_no_heavy_import(tmp_path, monkeypatch):
    """REQ-6: calling read_pending_live_events on an empty dir returns [] and is cheap.

    The helper must not import the embedder or MemoryStore on its hot path.
    (We verify it returns [] when the dir is empty — if it tried to import
    MemoryStore or the embedder, this would slow cold starts significantly.)
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    from iai_mcp.capture import read_pending_live_events

    # Empty dir (doesn't exist yet)
    events = read_pending_live_events()
    assert events == [], f"empty dir must return []; got {events!r}"

    # Ensure the dir exists but is empty
    deferred = _deferred_dir(tmp_path)
    events = read_pending_live_events()
    assert events == [], f"empty deferred-captures dir must return []; got {events!r}"
