"""
Platform-agnostic IPC transport layer.

POSIX:   Unix-domain socket  →  ~/.iai-mcp/.daemon.sock
Windows: TCP loopback         →  127.0.0.1:<ephemeral port>
         Port is persisted in ~/.iai-mcp/.daemon.port so clients can find it.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import platform
import socket
from pathlib import Path
from typing import Any

IS_WINDOWS: bool = platform.system() == "Windows"

_BASE_DIR: Path = Path.home() / ".iai-mcp"
SOCKET_PATH: Path = _BASE_DIR / ".daemon.sock"  # POSIX only — kept for compatibility
PORT_FILE: Path = _BASE_DIR / ".daemon.port"     # Windows only


# ---------------------------------------------------------------------------
# Port file helpers (Windows only)
# ---------------------------------------------------------------------------

def _read_port() -> int | None:
    try:
        return int(PORT_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _write_port(port: int) -> None:
    PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PORT_FILE.write_text(str(port), encoding="utf-8")


def _remove_port_file() -> None:
    try:
        PORT_FILE.unlink()
    except (FileNotFoundError, OSError):
        pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ipc_address() -> str | tuple[str, int]:
    """
    Return the current IPC endpoint.
    POSIX: Unix socket path string.
    Windows: ("127.0.0.1", port) tuple.
    """
    if not IS_WINDOWS:
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        return env if env else str(SOCKET_PATH)
    port = _read_port()
    if port is None:
        raise FileNotFoundError(
            "Daemon not running: ~/.iai-mcp/.daemon.port not found."
        )
    return ("127.0.0.1", port)


async def open_ipc_connection(
    addr: str | tuple[str, int] | None = None,
    *,
    timeout: float | None = None,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """
    Open a client connection to the daemon.

    On POSIX wraps asyncio.open_unix_connection; on Windows wraps
    asyncio.open_connection over TCP loopback.

    The *addr* parameter is ignored on Windows (always uses port file).
    """
    coro: Any
    if IS_WINDOWS:
        port = _read_port()
        if port is None:
            raise FileNotFoundError(
                "Daemon not running: ~/.iai-mcp/.daemon.port not found."
            )
        coro = asyncio.open_connection("127.0.0.1", port)
    else:
        if addr is None:
            env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
            addr = env if env else str(SOCKET_PATH)
        coro = asyncio.open_unix_connection(str(addr))

    if timeout is not None:
        return await asyncio.wait_for(coro, timeout=timeout)
    return await coro


async def start_ipc_server(
    handler: Any,
    addr: str | Path | None = None,
) -> tuple[asyncio.AbstractServer, str | tuple[str, int], bool]:
    """
    Start the daemon server.

    Returns ``(server, actual_addr, needs_manual_cleanup)`` where:
    - *actual_addr* is the socket path (POSIX) or ("127.0.0.1", port) (Windows).
    - *needs_manual_cleanup* is True if the caller must call ``shutdown_ipc``
      in its finally block (i.e. asyncio will NOT clean up automatically).

    On Windows the port is written to PORT_FILE immediately after bind.
    """
    if IS_WINDOWS:
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port: int = server.sockets[0].getsockname()[1]
        _write_port(port)
        return server, ("127.0.0.1", port), True

    # POSIX: try to use asyncio's built-in cleanup_socket (Python 3.12+)
    if addr is None:
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        path_str = env if env else str(SOCKET_PATH)
    else:
        path_str = str(addr)

    sig = inspect.signature(asyncio.start_unix_server)
    supports_cleanup = "cleanup_socket" in sig.parameters
    kwargs: dict[str, Any] = {"cleanup_socket": True} if supports_cleanup else {}

    server = await asyncio.start_unix_server(handler, path=path_str, **kwargs)
    return server, path_str, not supports_cleanup


def cleanup_ipc_address(addr: str | Path | None = None) -> None:
    """
    Remove a stale socket file before binding (POSIX only). No-op on Windows.
    """
    if IS_WINDOWS:
        return
    if addr is None:
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        path = Path(env) if env else SOCKET_PATH
    else:
        path = Path(addr)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


def shutdown_ipc(addr: str | tuple[str, int] | None = None) -> None:
    """
    Clean up after daemon shutdown.
    POSIX: unlink the socket file (idempotent).
    Windows: remove the port file.
    """
    if IS_WINDOWS:
        _remove_port_file()
        return
    if addr is None or isinstance(addr, tuple):
        env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
        path = Path(env) if env else SOCKET_PATH
    else:
        path = Path(addr)
    try:
        path.unlink()
    except (FileNotFoundError, OSError):
        pass


def make_sync_ipc_socket() -> tuple[socket.socket, str | tuple[str, int]]:
    """
    Create a synchronous (blocking) client socket and the address to connect to.

    Returns ``(sock, addr)`` where *addr* is a string path (POSIX) or
    ``("127.0.0.1", port)`` tuple (Windows).  Caller is responsible for
    ``settimeout``, ``connect``, and ``close``.
    """
    if IS_WINDOWS:
        port = _read_port()
        if port is None:
            raise FileNotFoundError(
                "Daemon not running: ~/.iai-mcp/.daemon.port not found."
            )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return s, ("127.0.0.1", port)

    env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
    path = env if env else str(SOCKET_PATH)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    return s, path
