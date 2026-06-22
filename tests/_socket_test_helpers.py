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
import socket
from pathlib import Path

from iai_mcp._ipc import IS_WINDOWS


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
