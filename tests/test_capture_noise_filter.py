"""_parse_transcript_line must drop hook/skill/system noise as role:user.

Preservation tests (test_genuine_line_preserved,
test_genuine_line_quoting_marker_preserved) confirm genuine lines survive.
"""
from __future__ import annotations

import json

import pytest

from iai_mcp.capture import _parse_transcript_line


def _user_line(text: str) -> str:
    """Build a valid Claude Code JSONL user-turn line with the given content."""
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}})


# ---- noise-class tests (RED today: no filter yet, all currently parse as user turns) ----


def test_command_message_dropped():
    """REQ-4: a <command-message> line must not be stored as a user turn."""
    line = _user_line("<command-message>some-command</command-message>")
    result = _parse_transcript_line(line)
    # RED today: filter absent, returns ("user", text) instead of None
    assert result is None, (
        f"command-message line should be filtered (got {result!r}); "
        "Plan 02 must add the noise filter to _parse_transcript_line"
    )


def test_skill_injection_dropped():
    """REQ-4: a skill-injection (Base directory for this skill:) must not be stored."""
    line = _user_line("Base directory for this skill: /Users/you/project")
    result = _parse_transcript_line(line)
    # RED today: filter absent
    assert result is None, (
        f"skill-injection line should be filtered (got {result!r}); "
        "Plan 02 must add the noise filter to _parse_transcript_line"
    )


def test_task_notification_dropped():
    """REQ-4: a <task-notification> line must not be stored as a user turn."""
    line = _user_line("<task-notification>\n<task-id>abc123</task-id>\n</task-notification>")
    result = _parse_transcript_line(line)
    # RED today: filter absent
    assert result is None, (
        f"task-notification line should be filtered (got {result!r}); "
        "Plan 02 must add the noise filter to _parse_transcript_line"
    )


def test_interrupted_dropped():
    """REQ-4: [Request interrupted by user] exact string must not be stored."""
    line = _user_line("[Request interrupted by user]")
    result = _parse_transcript_line(line)
    # RED today: filter absent
    assert result is None, (
        f"interrupted marker should be filtered (got {result!r}); "
        "Plan 02 must add the noise filter to _parse_transcript_line"
    )


# ---- preservation tests (GREEN today: genuine user turns parse correctly) ----


def test_genuine_line_preserved():
    """A genuine user line is stored byte-identical (no strip beyond input)."""
    # Feed already-stripped text so _parse_transcript_line's.strip() does not
    # change the value, making byte-identity verifiable.
    genuine_text = "what was the session identifier for the last worktree build"
    line = _user_line(genuine_text)
    result = _parse_transcript_line(line)
    assert result is not None, "genuine user line must not be filtered"
    role, text, *_ = result  # uuid/timestamp fields are None for simple fixtures
    assert role == "user"
    assert text == genuine_text, (
        f"MEM-01 violation: text was altered (got {text!r}, expected {genuine_text!r})"
    )


def test_genuine_line_quoting_marker_preserved():
    """boundary: a genuine user turn that CONTAINS a noise marker as a substring
    must NOT be dropped. Only prefix/exact matches are noise (startswith / ==).
    """
    genuine_text = "I saw <task-notification> appear in the logs yesterday"
    line = _user_line(genuine_text)
    result = _parse_transcript_line(line)
    assert result is not None, (
        "genuine user line containing a noise substring must not be filtered; "
        "MEM-01 requires byte-identical storage of real user turns"
    )
    role, text, *_ = result  # uuid/timestamp fields are None for simple fixtures
    assert role == "user"
    assert text == genuine_text
