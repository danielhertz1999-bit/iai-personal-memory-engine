"""RED: iai-mcp topology CLI.

The `topology` subcommand prints one key:value line per metric:
    C: <float>
    L: <float>
    sigma: <float | "insufficient_data">
    communities: <int>
    rich_club_ratio: <float>
    N: <int>
    regime: <str>
"""
from __future__ import annotations

import re

import pytest

from iai_mcp.cli import main as cli_main


def test_topology_subcommand_registered():
    """`iai-mcp topology --help` must succeed (subparser registered)."""
    with pytest.raises(SystemExit) as ex:
        cli_main(["topology", "--help"])
    # argparse --help calls sys.exit(0) on success
    assert ex.value.code == 0


def test_topology_prints_required_keys(tmp_path, capsys, monkeypatch):
    """All seven key:value lines must appear in output."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    code = cli_main(["topology"])
    assert code == 0
    out = capsys.readouterr().out

    assert re.search(r"^C:\s", out, re.MULTILINE), f"missing 'C: ' line in {out!r}"
    assert re.search(r"^L:\s", out, re.MULTILINE), f"missing 'L: ' line in {out!r}"
    assert re.search(r"^sigma:\s", out, re.MULTILINE), (
        f"missing 'sigma: ' line in {out!r}"
    )
    assert re.search(r"^communities:\s", out, re.MULTILINE), (
        f"missing 'communities: ' line in {out!r}"
    )
    assert re.search(r"^rich_club_ratio:\s", out, re.MULTILINE), (
        f"missing 'rich_club_ratio: ' line in {out!r}"
    )
    assert re.search(r"^N:\s", out, re.MULTILINE), f"missing 'N: ' line in {out!r}"
    assert re.search(r"^regime:\s", out, re.MULTILINE), (
        f"missing 'regime: ' line in {out!r}"
    )


def test_topology_empty_store_prints_insufficient_data(tmp_path, capsys, monkeypatch):
    """Fresh store: N is small, sigma should print as 'insufficient_data'."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    code = cli_main(["topology"])
    assert code == 0
    out = capsys.readouterr().out
    # On an empty store, sigma must be "insufficient_data" or the regime is
    # "insufficient_data" -- either way, the line must contain the marker.
    assert "insufficient_data" in out, (
        f"empty store must surface insufficient_data; got {out!r}"
    )
