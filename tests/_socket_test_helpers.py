"""Cross-platform fake-daemon socket binding for tests.

Production code reaches the daemon via ``iai_mcp._ipc``: on POSIX a unix-domain
socket at ``IAI_DAEMON_SOCKET_PATH``; on Windows TCP loopback with the port
persisted to ``"<IAI_DAEMON_SOCKET_PATH>.port"``. Tests that stand up a *raw*
fake daemon socket (to simulate stalls, fast replies, dead endpoints, etc.)
must bind the matching transport so the production client actually connects to
them. This helper hides the per-platform binding; callers keep their own
accept/recv/reply logic unchanged.
"""
from __future__ import annotations

import os
import secrets
import socket
from pathlib import Path

from iai_mcp._ipc import IS_WINDOWS


def write_fake_daemon_token(sock_path) -> None:
    """Write an auth token alongside a fake daemon socket so the production
    client's mandatory Windows handshake (see ``_ipc._send_token_async``) finds
    one. The raw fake servers don't validate it, so any value works. No-op on
    POSIX, where access control is the unix-socket file permissions."""
    if IS_WINDOWS:
        Path(f"{sock_path}.token").write_text(secrets.token_hex(16), encoding="utf-8")


def send_daemon_token(sock: socket.socket, sock_path) -> None:
    """Send the auth token as the first line on a *raw* client socket, matching
    the daemon's Windows handshake. Reads ``<sock_path>.token`` (written by the
    daemon or by ``write_fake_daemon_token``). No-op on POSIX."""
    if IS_WINDOWS:
        token = Path(f"{sock_path}.token").read_text(encoding="utf-8").strip()
        sock.sendall((token + "\n").encode("utf-8"))


def bind_fake_daemon_socket(sock_path) -> socket.socket:
    """Return a bound, listening socket that an ``_ipc`` client configured with
    ``IAI_DAEMON_SOCKET_PATH=sock_path`` will connect to.

    POSIX: ``AF_UNIX`` bound at ``sock_path``. Windows: ``AF_INET`` on
    ``127.0.0.1:<ephemeral>`` with the chosen port written to
    ``"<sock_path>.port"`` (matching ``_ipc._port_file_path``). Caller owns the
    returned socket (``accept``/``recv``/``close``).
    """
    if IS_WINDOWS:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        Path(f"{sock_path}.port").write_text(str(port), encoding="utf-8")
        write_fake_daemon_token(sock_path)
    else:
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(str(sock_path))
    srv.listen(5)
    return srv


def daemon_endpoint_ready_path(sock_path) -> Path:
    """Path that exists once a daemon bound at ``sock_path`` is reachable: the
    unix socket file on POSIX, the ``<sock_path>.port`` file on Windows."""
    return Path(f"{sock_path}.port") if IS_WINDOWS else Path(sock_path)


def daemon_endpoint(sock_path):
    """Connect target for a daemon bound at ``sock_path``: the unix socket path
    (POSIX) or ``("127.0.0.1", port)`` read from ``<sock_path>.port`` (Windows).
    Raises ``FileNotFoundError`` if the Windows port file is absent."""
    if IS_WINDOWS:
        port = int(Path(f"{sock_path}.port").read_text(encoding="utf-8").strip())
        return ("127.0.0.1", port)
    return str(sock_path)


def new_daemon_client_socket() -> socket.socket:
    """A raw client socket of the right family for the current platform
    (``AF_INET`` on Windows, ``AF_UNIX`` on POSIX)."""
    family = socket.AF_INET if IS_WINDOWS else socket.AF_UNIX
    return socket.socket(family, socket.SOCK_STREAM)
