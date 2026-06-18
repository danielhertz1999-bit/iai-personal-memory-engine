# Windows Port Handoff

## What this project is

`iai-personal-memory-engine` (repo at `C:\Users\Daniel Hertz\Documents\GitHub\iai-personal-memory-engine`)
is a local MCP server that gives Claude Code persistent long-term memory across sessions.
It captures every conversation, builds a personal model of the user, and injects relevant
context at session start — automatically. It is Python + Rust (PyO3), with a Node.js MCP wrapper.

It was macOS-only. We are porting it to Windows.

## What has already been done (Step 1 — committed)

**Commit:** `1dc1d64` — "Add platform-agnostic IPC transport layer for Windows porting"

Created `src/iai_mcp/_ipc.py` — a platform-agnostic IPC abstraction module.

- On POSIX: delegates to the existing Unix-domain socket at `~/.iai-mcp/.daemon.sock`
- On Windows: uses TCP loopback `127.0.0.1:<ephemeral port>`, port stored in `~/.iai-mcp/.daemon.port`

Updated all 9 callsites that previously used raw `asyncio.open_unix_connection` /
`asyncio.start_unix_server` / `socket.AF_UNIX`:
- `src/iai_mcp/concurrency.py`
- `src/iai_mcp/socket_server.py`
- `src/iai_mcp/cli/__init__.py`
- `src/iai_mcp/core/__init__.py`
- `src/iai_mcp/direct_write.py`
- `src/iai_mcp/daemon/_watchdog.py`
- `src/iai_mcp/doctor/_lifecycle_checks.py`
- `src/iai_mcp/doctor/__init__.py`
- `src/iai_mcp/semantic_recall.py`

## Completion Status

**Steps 1-6: COMPLETED** ✅

- **Step 1** (`1dc1d64`): Platform-agnostic IPC (Unix sockets → TCP loopback on Windows)
- **Step 2** (`8154b9b`): fcntl file locking → `_filelock.py` shim
- **Steps 3+4+9** (`c009736`): POSIX signals, resource module, CLI daemon logging
- **Steps 7+10** (`8ecd257`): uid/geteuid guards, os.fchmod guards, icacls file security
- **Step 5** (`0e8321c`): Windows Task Scheduler daemon installer (schtasks.exe)
- **Step 6** (`f4865bf`): PowerShell hook equivalents (.ps1 scripts + hook installer updates)

## What remains

Bench files (lower priority) and any final edge cases.

### Bench Files — resource.getrusage() (OPTIONAL — not required for daemon)

Lower priority, affects only benchmarking tools (not runtime code).

`fcntl` is POSIX-only. On Windows, importing any of these files raises `ModuleNotFoundError`.

Files to fix:
- `src/iai_mcp/capture_queue.py` — uses `fcntl.flock()`
- `src/iai_mcp/hippo/_db.py` — uses `fcntl.flock()`
- `src/iai_mcp/lifecycle_event_log.py` — uses `fcntl.flock()`
- `src/iai_mcp/lifecycle.py` — uses `fcntl.flock()`
- `src/iai_mcp/lock_protocol.py` — uses `fcntl.flock()`
- `src/iai_mcp/doctor/_lifecycle_checks.py` — uses `fcntl.flock()`

**Fix:** Create `src/iai_mcp/_filelock.py` that provides a `flock(fd, operation)` shim:
- On POSIX: delegates to `fcntl.flock(fd, operation)`
- On Windows: uses `msvcrt.locking()` with appropriate size (use `os.path.getsize` or a large constant like `2**31 - 1`)

Example shim:
```python
import platform, os
if platform.system() == "Windows":
    import msvcrt
    LOCK_EX = 1; LOCK_SH = 2; LOCK_UN = 4; LOCK_NB = 8
    def flock(fd, operation):
        if isinstance(fd, int):
            raw = fd
        else:
            raw = fd.fileno()
        if operation & LOCK_UN:
            try: msvcrt.locking(raw, msvcrt.LK_UNLCK, 2**30)
            except OSError: pass
        elif operation & LOCK_EX:
            mode = msvcrt.LK_NBLCK if (operation & LOCK_NB) else msvcrt.LK_LOCK
            msvcrt.locking(raw, mode, 2**30)
        elif operation & LOCK_SH:
            mode = msvcrt.LK_NBLCK if (operation & LOCK_NB) else msvcrt.LK_LOCK
            msvcrt.locking(raw, mode, 2**30)
else:
    import fcntl as _fcntl
    LOCK_EX = _fcntl.LOCK_EX; LOCK_SH = _fcntl.LOCK_SH
    LOCK_UN = _fcntl.LOCK_UN; LOCK_NB = _fcntl.LOCK_NB
    def flock(fd, operation):
        _fcntl.flock(fd, operation)
```

Then in each affected file, replace:
```python
import fcntl
...
fcntl.flock(fd, fcntl.LOCK_EX)
```
with:
```python
from iai_mcp._filelock import flock, LOCK_EX, LOCK_SH, LOCK_UN, LOCK_NB
...
flock(fd, LOCK_EX)
```

---

### Step 3 — resource module (CRITICAL — daemon crashes on import)

`resource` is POSIX-only. `src/iai_mcp/daemon/__init__.py` imports it at the top level.

Files to fix:
- `src/iai_mcp/daemon/__init__.py` — `resource.getrlimit()`, `resource.setrlimit()`

**Fix:** Wrap in a platform guard:
```python
import platform as _platform
if _platform.system() != "Windows":
    import resource as _resource
    def _raise_fd_limit():
        soft, hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        if soft < 4096:
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (min(4096, hard), hard))
else:
    def _raise_fd_limit():
        pass  # Windows manages FD limits via OS handles
```

Also fix in bench files (lower priority, bench-only):
- `bench/memory_footprint.py`, `bench/embed_warm_cost.py`, `bench/consolidation_rss_peak.py`,
  `bench/memorygraph_memory.py` — use `psutil.Process(os.getpid()).memory_info().rss` instead
  of `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`

---

### Step 4 — POSIX signals (CRITICAL — daemon crashes on Windows)

`signal.SIGHUP`, `signal.SIGKILL` do not exist on Windows.

Files to fix:
- `src/iai_mcp/daemon/__init__.py` — registers SIGHUP handler; calls SIGTERM/SIGKILL
- `src/iai_mcp/daemon/_watchdog.py` — `os.kill(os.getpid(), signal.SIGKILL)`
- `src/iai_mcp/cli/_daemon.py` — `os.kill(pid, signal.SIGTERM)` / `SIGKILL`
- `src/iai_mcp/doctor/__init__.py` — `os.kill(pid, signal.SIGTERM)`

**Fix:**
```python
import platform, signal, os

def _terminate_process(pid: int, graceful: bool = True) -> None:
    if platform.system() == "Windows":
        os.kill(pid, signal.CTRL_C_EVENT)
    else:
        sig = signal.SIGTERM if graceful else signal.SIGKILL
        os.kill(pid, sig)

# For SIGHUP registration, guard it:
if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, _reload_handler)
```

For `os.kill(os.getpid(), signal.SIGKILL)` (self-termination in watchdog), replace with
`sys.exit(1)` on Windows.

---

### Step 5 — Daemon installer: Windows Task Scheduler (MAJOR)

`iai-mcp daemon install` only supports launchd (macOS) and systemd (Linux).
It needs a Windows backend.

File: `src/iai_mcp/cli/_daemon.py`

Add `_is_windows()` guard and implement `cmd_daemon_install_windows()` that:
1. Uses Python's `subprocess` to call `schtasks.exe` — the built-in Windows Task Scheduler CLI.
2. Creates a task that runs `pythonw.exe -m iai_mcp.daemon` at login, hidden.
3. Writes a `WINDOWS_SERVICE_TARGET` path constant analogous to `LAUNCHD_TARGET`.

Example schtasks command:
```
schtasks /Create /SC ONLOGON /TN "iai-mcp-daemon" /TR "pythonw.exe -m iai_mcp.daemon" /RL HIGHEST /F
```

Also implement `cmd_daemon_uninstall_windows()`:
```
schtasks /Delete /TN "iai-mcp-daemon" /F
```

And `cmd_daemon_start_windows()` / `cmd_daemon_stop_windows()`:
```
schtasks /Run /TN "iai-mcp-daemon"
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq iai-mcp-daemon"
```

Wire these into the existing `cmd_daemon_install()` dispatch block alongside the
`_is_macos()` and `_is_linux()` branches.

---

### Step 6 — Shell hooks: PowerShell equivalents (MAJOR)

Claude Code on Windows does not run `.sh` hook scripts. The three hooks need `.ps1` equivalents.

Hooks are in `src/iai_mcp/_deploy/hooks/`:
- `iai-mcp-turn-capture.sh` — appends each prompt+response turn to per-session buffer
- `iai-mcp-session-capture.sh` — at session end, rolls the buffer for the daemon
- `iai-mcp-session-recall.sh` — at session start, pipes cached memory prefix to stdout

**Fix:** Create `.ps1` versions of each that call the Python CLI equivalents:
```powershell
# iai-mcp-turn-capture.ps1
$python = (Get-Command python).Source
& $python -m iai_mcp capture-turn @args
```
The Python CLI already has `capture-transcript`, `session-start` subcommands —
the PowerShell hooks just need to call them.

Also update `src/iai_mcp/cli/_capture.py`'s `cmd_capture_hooks_install()` to:
1. Detect Windows and copy `.ps1` files instead of `.sh` files
2. Patch `~/.claude/settings.json` hooks to reference `.ps1` paths on Windows

---

### Step 7 — os.getuid / pwd module guards (MODERATE)

`os.getuid()` and the `pwd` module are POSIX-only.

Files to fix:
- `src/iai_mcp/crypto.py` — `os.geteuid()` at line ~121
- `src/iai_mcp/cli/_crypto.py` — `st.st_uid == os.geteuid()` at line ~39
- `src/iai_mcp/hippo/__init__.py` — `pwd.getpwuid(os.getuid()).pw_dir` at line ~54

**Fix:**
```python
# For ownership checks:
if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
    raise PermissionError(...)

# For home directory (hippo/__init__.py):
# Replace pwd.getpwuid(os.getuid()).pw_dir with:
home = str(Path.home())
```

---

### Step 8 — Rust build: disable macOS-only features (MODERATE)

`rust/iai_mcp_embed_core/Cargo.toml` has `accelerate` and `metal` features
(Apple Accelerate framework and Apple Metal GPU). These fail to compile on Windows.

**Fix:** In `pyproject.toml` (the setuptools-rust build config), add platform-conditional
feature flags. Find the `[[tool.setuptools-rust.ext-modules]]` section and add:

```toml
[[tool.setuptools-rust.ext-modules]]
target = "iai_mcp_native"
path = "rust/iai_mcp_native/Cargo.toml"
binding = "PyO3"
features = ["extension-module"]
args = ["--no-default-features"]
```

This already disables default features. Verify `accelerate` and `metal` are not in the
default feature set of `Cargo.toml`. If they are, add a `[target.'cfg(target_os = "macos")'.dependencies]`
section in `Cargo.toml` to gate them.

---

### Step 9 — Log paths and temp dirs (MINOR)

`src/iai_mcp/cli/_daemon.py` uses `~/Library/Logs/` for daemon logs (macOS-specific).

**Fix:** Add `_get_daemon_log_path()`:
```python
import platform
def _get_daemon_log_path() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Logs" / "iai-mcp-daemon.stderr.log"
    elif platform.system() == "Windows":
        return Path(os.environ.get("APPDATA", Path.home())) / "iai-mcp" / "logs" / "daemon.log"
    else:
        return Path.home() / ".local" / "share" / "iai-mcp" / "logs" / "daemon.log"
```

---

### Step 10 — chmod security for crypto key (MINOR)

`src/iai_mcp/crypto.py` calls `os.chmod(key_file, 0o600)` to restrict the encryption key.
On Windows, `chmod` is a no-op for access control. Use `icacls.exe` instead:

```python
import platform, subprocess
def _secure_key_file(path: Path) -> None:
    if platform.system() == "Windows":
        user = os.environ.get("USERNAME", "")
        subprocess.run(
            ["icacls", str(path), "/inheritance:d", "/grant:r", f"{user}:F"],
            check=False, capture_output=True,
        )
    else:
        path.chmod(0o600)
```

---

## Next Steps (for the next session)

The core daemon + hook infrastructure is now Windows-ready. Remaining work:

1. **Bench files (OPTIONAL, lower priority):** Update bench files that use `resource.getrusage()` to use `psutil.Process().memory_info().rss` instead. Affects:
   - `bench/memory_footprint.py`
   - `bench/embed_warm_cost.py`
   - `bench/consolidation_rss_peak.py`
   - `bench/memorygraph_memory.py`

2. **Manual testing on Windows:** Verify the port works by:
   ```powershell
   cd "C:\Users\Daniel Hertz\Documents\GitHub\iai-personal-memory-engine"
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev]"
   python -m iai_mcp daemon install --dry-run  # Check schtasks XML renders
   python -m iai_mcp capture-hooks install --dry-run  # Check hook paths
   ```

3. **Update CLAUDE.md:** Add Windows-specific setup notes to the project's CLAUDE.md (if it exists) or create one with:
   - Running `iai-mcp daemon install` on Windows (uses Task Scheduler)
   - Running `iai-mcp capture-hooks install` on Windows (uses PowerShell hooks)
   - Expected log locations (`%APPDATA%\iai-mcp\logs\`)

## Verification Checklist

After all steps complete:
- [ ] Daemon imports without crashing on Windows
- [ ] `iai-mcp daemon install` creates a Task Scheduler entry
- [ ] `iai-mcp capture-hooks install` creates PowerShell hooks and registers in settings.json
- [ ] Hook commands reference `.ps1` files (not `.sh`) on Windows in settings.json
- [ ] Logs go to `%APPDATA%\iai-mcp\logs\` (Windows) not `~/.local/share` (Linux)
- [ ] Crypto key file created with appropriate icacls permissions

## Key Design Decisions

1. **Platform detection:** Uses `platform.system()` checks (`== "Windows"`, `== "Darwin"`, `== "Linux"`) throughout
2. **File locking:** `_filelock.py` shim normalizes `msvcrt.locking()` (Windows) to `fcntl.flock()` interface (POSIX)
3. **Daemon management:** Task Scheduler on Windows, launchd on macOS, systemd on Linux
4. **Hooks:** Python calls wrapped in shell scripts (.sh on POSIX) or PowerShell scripts (.ps1 on Windows)
5. **No cross-platform abstractions:** Branching logic is explicit per-platform to avoid accidental breakage

After Step 5 (daemon installer):
```powershell
iai-mcp daemon install
iai-mcp daemon status
```

After Step 6 (hooks):
```powershell
iai-mcp capture-hooks install
iai-mcp capture-hooks status
```

Full E2E after all steps:
```powershell
iai-mcp doctor
```

## Notes

- The user is on Windows 11 Pro, Python 3.12, Node 18+, has Rust toolchain
- GitHub user: `danielhertz1999-bit`, repo fork is under their account
- The upstream repo is `CodeAbra/iai-personal-memory-engine`
- All changes should be committed to the local `main` branch; a PR to upstream can be opened later
- Keep each step as a separate commit for clean history
- The `setproctitle` module (used in `daemon/__init__.py`) may need a try/except fallback
  on Windows if it fails to compile — wrap: `try: from setproctitle import setproctitle\nexcept ImportError: setproctitle = lambda x: None`
