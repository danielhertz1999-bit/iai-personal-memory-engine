from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr


def _make_args(question: str = "test question", limit: int = 5):
    import argparse
    ns = argparse.Namespace()
    ns.question = question
    ns.limit = limit
    return ns


def _stub_recall_hits(monkeypatch, hits: list[dict]):
    fake_resp = {"jsonrpc": "2.0", "id": 1, "result": {"hits": hits}}
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)


def test_ask_happy_path_prints_answer_and_sources(monkeypatch):
    from iai_mcp import iai_cli

    _stub_recall_hits(monkeypatch, [
        {"id": "11111111-aaaa-bbbb-cccc-000000000001", "literal_surface": "fact one"},
        {"id": "22222222-aaaa-bbbb-cccc-000000000002", "literal_surface": "fact two"},
    ])

    def _fake_sync(prompt, **kwargs):
        assert "fact one" in prompt
        assert "fact two" in prompt
        return {
            "ok": True,
            "data": {"result": "Based on the memories, the answer is X."},
            "cost_usd": 0.0,
            "tokens_in": 100,
            "tokens_out": 30,
        }

    monkeypatch.setattr("iai_mcp.claude_cli.invoke_claude_sync", _fake_sync)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_ask(_make_args(question="what is X?", limit=5))

    assert rc == 0
    out = buf.getvalue()
    assert "Based on the memories" in out
    assert "Sources:" in out
    assert "11111111" in out
    assert "22222222" in out


def test_ask_no_hits_returns_nonzero_with_explanation(monkeypatch):
    from iai_mcp import iai_cli

    _stub_recall_hits(monkeypatch, [])

    called: list[bool] = []

    def _fake_sync(*a, **k):
        called.append(True)
        return {"ok": True, "data": {"result": "should not be invoked"}}

    monkeypatch.setattr("iai_mcp.claude_cli.invoke_claude_sync", _fake_sync)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_ask(_make_args())

    assert rc == 1
    assert "no memories" in err.getvalue().lower()
    assert not called, "claude_cli must NOT be invoked when there are 0 hits"


def test_ask_subscription_gate_denies(monkeypatch):
    from iai_mcp import iai_cli

    _stub_recall_hits(monkeypatch, [
        {"id": "abc", "literal_surface": "hello"},
    ])

    def _fake_sync(*a, **k):
        return {"ok": False, "reason": "credentials_file_missing"}

    monkeypatch.setattr("iai_mcp.claude_cli.invoke_claude_sync", _fake_sync)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_ask(_make_args())

    assert rc == 1
    assert "credentials_file_missing" in err.getvalue()


def test_ask_empty_answer_returns_nonzero(monkeypatch):
    from iai_mcp import iai_cli

    _stub_recall_hits(monkeypatch, [
        {"id": "abc", "literal_surface": "hello"},
    ])

    monkeypatch.setattr(
        "iai_mcp.claude_cli.invoke_claude_sync",
        lambda *a, **k: {"ok": True, "data": {"result": ""}},
    )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_ask(_make_args())

    assert rc == 1
    assert "empty answer" in err.getvalue().lower()


def test_ask_truncates_memories_to_limit(monkeypatch):
    from iai_mcp import iai_cli

    hits = [
        {"id": f"id-{i}", "literal_surface": f"memory {i}"}
        for i in range(10)
    ]
    _stub_recall_hits(monkeypatch, hits)

    seen_prompt: dict = {}

    def _fake_sync(prompt, **kwargs):
        seen_prompt["prompt"] = prompt
        return {"ok": True, "data": {"result": "answer"}}

    monkeypatch.setattr("iai_mcp.claude_cli.invoke_claude_sync", _fake_sync)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        iai_cli.cmd_ask(_make_args(limit=3))

    assert "memory 0" in seen_prompt["prompt"]
    assert "memory 1" in seen_prompt["prompt"]
    assert "memory 2" in seen_prompt["prompt"]
    assert "memory 3" not in seen_prompt["prompt"]


def test_ask_subcommand_registered():
    from iai_mcp.iai_cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["ask", "what is the answer to life?"])
    assert args.cmd == "ask"
    assert args.question == "what is the answer to life?"
    assert args.limit == 5
