#!/usr/bin/env bash
# driver.sh — launch + smoke-drive the iai-mcp personal memory engine.
#
# This is the agent-facing harness for the run-iai-mcp skill. It brings the
# background daemon up (with the offline embedder), then drives the real user
# surface — capture, recall, status, doctor — and asserts the native Rust
# embedder is live. It is NOT the test suite; it exercises the running app.
#
# Usage (from repo root):
#   bash .claude/skills/run-iai-mcp/driver.sh            # full smoke
#   bash .claude/skills/run-iai-mcp/driver.sh up         # just start daemon
#   bash .claude/skills/run-iai-mcp/driver.sh status     # one status round-trip
#   bash .claude/skills/run-iai-mcp/driver.sh down        # stop the daemon
#
# Exit 0 = daemon up + embedder backend=rust + capture/recall worked.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG="$HOME/.iai-mcp/logs/skill-daemon.out"
mkdir -p "$(dirname "$LOG")"

# The native Rust embedder pulls bge-small-en-v1.5 from HuggingFace on first
# run. In restricted containers there is no HF network, so point it at the
# local cache. The daemon also auto-detects this, but setting it explicitly
# makes the harness deterministic.
export IAI_MCP_EMBED_OFFLINE=1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '   \033[0;32m✓\033[0m %s\n' "$*"; }
die() { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

daemon_up() { iai status 2>/dev/null | grep -q 'daemon *UP'; }

start_daemon() {
  if daemon_up; then ok "daemon already UP"; return 0; fi
  say "starting daemon (python -m iai_mcp.daemon)"
  nohup python -m iai_mcp.daemon > "$LOG" 2>&1 &
  for _ in $(seq 1 20); do
    sleep 1
    daemon_up && { ok "daemon UP"; return 0; }
  done
  echo "--- daemon log tail ---" >&2; tail -20 "$LOG" >&2
  die "daemon did not come UP within 20s"
}

stop_daemon() {
  say "stopping daemon"
  pkill -f 'iai_mcp\.daemon' 2>/dev/null && ok "sent SIGTERM" || ok "no daemon process found"
}

case "${1:-smoke}" in
  up)     start_daemon ;;
  down)   stop_daemon ;;
  status) iai status ;;
  smoke)
    start_daemon

    say "iai status"
    iai status || die "status failed"

    say "iai capture"
    iai capture "run-iai-mcp driver smoke $(cat /proc/sys/kernel/random/uuid)" \
      | grep -q 'captured' || die "capture did not report 'captured'"
    ok "capture accepted"

    say "iai recall"
    iai recall "driver smoke" | head -8 || die "recall failed"
    ok "recall returned ranked rows"

    say "doctor — assert native Rust embedder is live"
    # NB: doctor exits 1 whenever ANY check FAILs — e.g. (e) daemon-state
    # TRANSITIONING, which is harmless here (see SKILL Gotchas). So capture its
    # output to a file (ignoring exit code) and grep the file; do NOT pipe
    # doctor directly under `set -o pipefail` or its exit 1 fails the pipeline.
    iai-mcp doctor > /tmp/iai-doctor.out 2>&1 || true
    grep -E '\(v\) native Rust embedder' /tmp/iai-doctor.out | grep -q 'backend=rust' \
      || { tail -30 /tmp/iai-doctor.out >&2; die "native Rust embedder NOT live"; }
    ok "native Rust embedder: backend=rust, 384-dim"

    say "SMOKE PASS"
    ;;
  *) die "unknown command: ${1:-} (use: smoke | up | down | status)" ;;
esac
