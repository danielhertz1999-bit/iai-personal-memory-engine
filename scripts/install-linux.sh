#!/usr/bin/env bash
# scripts/install-linux.sh — Fedora / RHEL / Debian Linux setup for iai-mcp.
#
# Usage (from repo root or anywhere inside the clone):
#   bash scripts/install-linux.sh
#
# Does:
#   1. checks prerequisites (Python, Rust, Node.js)
#   2. creates .venv if missing
#   3. installs iai-mcp editable into the venv
#   4. builds the TS MCP wrapper
#   5. symlinks ~/.local/bin/iai-mcp -> .venv/bin/iai-mcp
#   6. initializes crypto key
#   7. installs systemd user daemon service
#
# Idempotent. Safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[0;33m! \033[0m%s\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
step "prerequisites"

# Python (>=3.11)
PYTHON=""
for ver in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$ver" >/dev/null 2>&1; then
        pv=$("$ver" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
        minor=$(echo "$pv" | cut -d. -f2)
        if [ "${minor:-0}" -ge 11 ]; then
            PYTHON="$ver"
            break
        fi
    fi
done
[ -n "$PYTHON" ] || die "Python >=3.11 not found on PATH"
ok "$PYTHON ($( $PYTHON --version 2>&1 ))"

# Rust toolchain
if ! command -v rustc >/dev/null 2>&1; then
    warn "Rust not found — installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
ok "rustc ($(rustc --version | cut -d' ' -f2))"

# Node.js (>=18)
if ! command -v node >/dev/null 2>&1; then
    die "Node.js >=18 not found. Install via dnf install nodejs or nvm."
fi
ok "node $(node --version)"

# ---------------------------------------------------------------------------
# venv + pip install
# ---------------------------------------------------------------------------
step "python venv"
if [ ! -d .venv ]; then
    $PYTHON -m venv .venv
    ok ".venv created with $PYTHON"
else
    ok ".venv already exists"
fi

step "editable install (pip -e .)"
.venv/bin/pip install --quiet --upgrade pip setuptools wheel
.venv/bin/pip install --quiet -e "."
ok "iai-mcp python package installed into venv (Rust native extension built)"

# ---------------------------------------------------------------------------
# TS wrapper build
# ---------------------------------------------------------------------------
step "TS MCP wrapper"
if [ -d mcp-wrapper ]; then
    pushd mcp-wrapper >/dev/null
    if [ -f package-lock.json ]; then
        npm ci --silent --no-audit --no-fund
    else
        npm install --silent --no-audit --no-fund
    fi
    npm run build --silent
    popd >/dev/null
    ok "mcp-wrapper/dist built"
else
    warn "mcp-wrapper/ missing — skipping"
fi

# ---------------------------------------------------------------------------
# Global symlink into ~/.local/bin
# ---------------------------------------------------------------------------
step "global CLI symlink"
LOCAL_BIN="${HOME}/.local/bin"
LINK_PATH_IAMCP="${LOCAL_BIN}/iai-mcp"
LINK_PATH_IAI="${LOCAL_BIN}/iai"
TARGET_IAMCP="${REPO_ROOT}/.venv/bin/iai-mcp"
TARGET_IAI="${REPO_ROOT}/.venv/bin/iai"

mkdir -p "${LOCAL_BIN}"

for link in "$LINK_PATH_IAMCP" "$LINK_PATH_IAI"; do
    target="$([[ "$link" == *iamcp* ]] && echo "$TARGET_IAMCP" || echo "$TARGET_IAI")"
    if [ ! -x "$target" ]; then
        warn "entry point not found at $target — skipping symlink for $(basename "$link")"
        continue
    fi
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        die "$link exists and is NOT a symlink. move it aside and re-run."
    fi
    ln -sf "$target" "$link"
    ok "$link -> $target"
done

# PATH check
PATH_HAS_LOCAL_BIN="(.venv/bin/python -c "import os; print('1' if '${LOCAL_BIN}' in os.environ.get('PATH', '').split(':') else '0')")"
if [ "${PATH_HAS_LOCAL_BIN}" != "1" ]; then
    warn "${LOCAL_BIN} is NOT in your PATH"
    warn "add this to ~/.bashrc or ~/.zshrc and restart your shell:"
    warn "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
else
    ok "${LOCAL_BIN} is in PATH"
fi

# ---------------------------------------------------------------------------
# Crypto key initialization
# ---------------------------------------------------------------------------
step "crypto key"
if [ ! -f "${HOME}/.iai-mcp/.crypto.key" ] && [ -z "${IAI_MCP_CRYPTO_PASSPHRASE:-}" ]; then
    if "${REPO_ROOT}/.venv/bin/iai-mcp" crypto init >/dev/null 2>&1; then
        ok "crypto key generated (~/.iai-mcp/.crypto.key)"
    else
        warn "crypto init failed — run \`iai-mcp crypto init\` manually"
    fi
else
    ok "crypto key already present"
fi

# ---------------------------------------------------------------------------
# systemd user daemon service
# ---------------------------------------------------------------------------
step "systemd user daemon service"
if "${REPO_ROOT}/.venv/bin/iai-mcp" daemon install --yes 2>&1; then
    systemctl --user daemon-reload >/dev/null 2>&1 || true
    ok "systemd user service installed and enabled"
else
    warn "daemon install failed — run \`iai-mcp daemon install\` manually"
fi

# ---------------------------------------------------------------------------
# Enable linger (survive logout)
# ---------------------------------------------------------------------------
step "loginctl linger"
USER="${USER:-$(whoami)}"
if loginctl show-user "$USER" --property=Linger 2>/dev/null | grep -q "Linger=yes"; then
    ok "linger already enabled — daemon survives logout"
else
    if loginctl enable-linger "$USER" >/dev/null 2>&1; then
        ok "linger enabled for $USER"
    else
        warn "could not enable linger — daemon may die at logout"
    fi
fi

# ---------------------------------------------------------------------------
# Start the daemon
# ---------------------------------------------------------------------------
step "start daemon"
if systemctl --user start iai-mcp-daemon.service 2>/dev/null; then
    sleep 1
    if systemctl --user is-active iai-mcp-daemon.service >/dev/null 2>&1; then
        ok "daemon started and running (PID $(systemctl --user show -p MainPID iai-mcp-daemon.service --value))"
    else
        warn "daemon failed to start — check logs with: journalctl --user -u iai-mcp-daemon.service"
    fi
else
    warn "could not start daemon via systemctl (may need systemd)"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step "done"
echo ""
echo "   next steps:"
echo "     iai-mcp doctor              # verify installation"
echo "     iai-mcp capture-hooks install  # wire up Claude Code hooks"
echo "     claude mcp add iai-mcp -- node \"$(pwd)/mcp-wrapper/dist/index.js\""
echo ""
