"""Startup guard for the mandatory Rust native extension.

Call _require_native() at the top of every process entry point (daemon, stdio
MCP) to fail loud with an actionable build command when the native module is
absent or broken. This guard must NOT be imported from __init__.py -- that
would break test collection for tests that import iai_mcp without the wheel.
"""
from __future__ import annotations


def _require_native() -> None:
    """Verify that iai_mcp_native is importable and functional.

    Raises RuntimeError with a build command if the module is absent, broken,
    or missing required submodules. Succeeds silently when native is present
    and healthy.

    Import discipline: this module imports only iai_mcp_native and its
    submodules. It must never import any other iai_mcp module -- doing so
    would create a circular import and undermine the guard's ability to
    report a missing native cleanly.
    """
    try:
        import iai_mcp_native  # noqa: F401
        from iai_mcp_native import embed as _e, graph as _g  # touch both submodules

        # Smoke-call into each submodule: a present-but-corrupt.so raises on
        # attribute access, proving the extension is fully initialised and not
        # just partially linked.
        assert callable(_g.is_connected), "iai_mcp_native.graph.is_connected not callable"
        assert callable(_e.Embedder), "iai_mcp_native.embed.Embedder not callable"
    except Exception as exc:
        raise RuntimeError(
            "iai_mcp_native (the Rust native extension) is not available.\n"
            "\n"
            "The Rust embedder and graph algorithms have no Python fallback.\n"
            "A missing or broken native module prevents the daemon and the\n"
            "MCP server from starting.\n"
            "\n"
            "To rebuild the extension, run:\n"
            "\n"
            "    iai-mcp build-native\n"
            "\n"
            "Then restart the daemon or MCP server.\n"
            "\n"
            "Advanced / manual build from a source checkout:\n"
            "\n"
            "    cd rust/iai_mcp_native && maturin develop --release"
        ) from exc
