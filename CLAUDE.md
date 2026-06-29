# iai-personal-memory-engine — Developer Notes for Claude

## What this project is

A local MCP server that gives Claude Code persistent cross-session memory.
It captures every conversation turn, builds a personal model of the user, and
injects relevant context at the start of each new session automatically.

Stack: Python 3.11/3.12 + Rust (PyO3) + Node.js MCP wrapper.

## Quick orientation

```
src/iai_mcp/          # Python package (runtime, CLI, daemon, doctor)
rust/                 # Native Rust extension (embedder, graph kernels)
mcp-wrapper/          # Node.js MCP transport wrapper
bench/                # Benchmarking scripts
tests/                # Pytest test suite
scripts/              # install.sh / install.ps1 / uninstall.*
```

Key entry points:
- `src/iai_mcp/cli/__init__.py` — `iai-mcp` CLI dispatcher
- `src/iai_mcp/daemon/__init__.py` — async background daemon
- `src/iai_mcp/_ipc.py` — platform-agnostic IPC (Unix socket / Windows TCP)
- `src/iai_mcp/_filelock.py` — platform-agnostic file locking (fcntl / msvcrt)

## Platform support

### macOS / Linux

Standard path. See `README.md` → Quick start.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
cd mcp-wrapper && npm install && npm run build && cd ..
iai-mcp daemon install   # launchd on macOS, systemd on Linux
iai-mcp capture-hooks install
iai-mcp doctor
```

### Windows

The Windows port is complete. Use the PowerShell installer:

```powershell
# From repo root in PowerShell:
.\scripts\install.ps1
```

This script:
1. Creates `.venv` with Python 3.11 or 3.12
2. Runs `pip install -e .` (builds the Rust native engine via setuptools-rust + MSVC)
3. Builds `mcp-wrapper\dist` via `npm run build`
4. Adds `.venv\Scripts` to the user PATH
5. Installs the daemon via **Windows Task Scheduler** (runs `pythonw.exe -m iai_mcp.daemon` at logon)
6. Optionally installs ambient capture/recall hooks

**Prerequisites for Windows:**
- Python 3.11 or 3.12 from python.org (check "Add to PATH")
- Node.js 18+ from nodejs.org
- Rust toolchain from rustup.rs
- Visual C++ Build Tools (cargo needs the MSVC linker — install via `winget install Microsoft.VisualStudio.2022.BuildTools`)

#### Windows daemon management

```powershell
iai-mcp daemon install    # register Task Scheduler task (starts at logon)
iai-mcp daemon start      # run task now: schtasks /Run /TN iai-mcp-daemon
iai-mcp daemon stop       # end task: schtasks /End /TN iai-mcp-daemon
iai-mcp daemon uninstall  # delete task: schtasks /Delete /TN iai-mcp-daemon /F
iai-mcp daemon status     # check: schtasks /Query /TN iai-mcp-daemon
iai-mcp daemon logs       # open %APPDATA%\iai-mcp\logs\ in Explorer
```

#### Windows log location

```
%APPDATA%\iai-mcp\logs\daemon.log
```

#### Windows IPC security

On POSIX the daemon uses a Unix-domain socket (filesystem ACLs provide access control).
On Windows it uses TCP loopback `127.0.0.1:<ephemeral port>`. Because any local process
can connect to a TCP loopback port, an auth-token handshake is layered on top:

- Daemon generates a 32-byte random token at startup, writes it to `~/.iai-mcp/.daemon.token`
  with `icacls` ACL-restricted to the current user.
- Every client sends the token as the first line of each connection.
- Connections with the wrong token are closed immediately (`secrets.compare_digest`).

The token is regenerated on every daemon start and removed on shutdown.

#### Windows file locking

`fcntl.flock` is POSIX-only. All callers use the `iai_mcp._filelock` shim instead:

```python
from iai_mcp._filelock import flock, LOCK_EX, LOCK_SH, LOCK_NB, LOCK_UN
```

On Windows this delegates to `msvcrt.locking`. Known divergence: `LOCK_SH` is
serviced as an exclusive lock on Windows (two concurrent readers block each other).
This is a throughput limitation, not a correctness one — see the docstring in
`src/iai_mcp/_filelock.py` for the full rationale.

#### Windows capture/recall hooks

Claude Code on Windows runs `.ps1` hooks, not `.sh` hooks. The installer deploys
PowerShell equivalents automatically. They call the same Python CLI under the hood.

```powershell
iai-mcp capture-hooks install   # deploys .ps1 hooks + patches ~/.claude/settings.json
iai-mcp capture-hooks status    # shows hook paths and registration state
```

#### Windows uninstall

```powershell
.\scripts\uninstall.ps1              # remove Task Scheduler task + kill daemon
.\scripts\uninstall.ps1 -PurgeState  # also remove daemon state files
.\scripts\uninstall.ps1 -PurgeData   # also delete memory store (DESTRUCTIVE)
```

## Running tests

```bash
pytest tests/ -x -q                      # fast subset (no slow marker)
pytest tests/ -x -q -m "not slow"        # explicit fast-only
pytest tests/ -x -q -m slow              # integration / subprocess tests
```

The test suite is cross-platform. Tests that rely on POSIX paths or `/tmp` use
`tmp_path` fixtures. Tests that assert `os.stat` mode bits skip on Windows.

## Useful diagnostics

```bash
iai-mcp doctor               # 25-point health check
iai-mcp doctor --apply --yes # auto-repair common issues
iai capture 'hello'          # test memory write (prints: captured id=...)
iai recall 'hello'           # test recall (prints ranked results)
iai status                   # daemon UP + record count
```

## Offline embedder (container / restricted environments)

The native Rust embedder downloads `bge-small-en-v1.5` from HuggingFace on first run.
In environments without HuggingFace network access, download the model first via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-small-en-v1.5")
```

Then set `IAI_MCP_EMBED_OFFLINE=1` before starting the daemon, or let the daemon
auto-detect the cached model (it sets the env var automatically if the cache exists).

## Architecture notes

- The daemon holds an exclusive SQLite write lock on `~/.iai-mcp/hippo/brain.sqlite3`.
  Only one daemon instance can run at a time.
- Embeddings are 384-dimensional float32 (bge-small-en-v1.5).
- The FAISS/hnswlib ANN index lives alongside the SQLite store.
- Memory tiers: `episodic` (raw turns), `semantic` (consolidated facts), `identity` (stable traits).
- Community detection runs the MOSAIC algorithm (pure Python, MIT license).
