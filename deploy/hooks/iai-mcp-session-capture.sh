#!/usr/bin/env bash
# IAI-MCP Stop hook — ambient WRITE-side capture (Plan 06 + Phase 7.1).
#
# Fires when a Claude Code session ends. Reads the session's JSONL transcript,
# batch-captures user + assistant turns into the iai-mcp episodic tier through
# `iai-mcp capture-transcript --no-spawn`. NEVER spawns a daemon (Phase 7.1 R3).
# If the daemon is unreachable, the call defers events to
# ~/.iai-mcp/.deferred-captures/ for the daemon to drain on next socket
# activation (handled by drain_deferred_captures in daemon.main + _tick_body
# WAKE handler — Plan 07.1-06).
#
# Fail-safe by design: any error exits 0 so session teardown is never blocked.
# Logs go to ~/.iai-mcp/logs/capture-YYYY-MM-DD.log for audit.
#
# Hook payload (stdin JSON from Claude Code) contains:
#   - session_id       (UUID of the session that just ended)
#   - transcript_path  (absolute path to the session JSONL) — available in
#                      newer Claude Code builds; we fall back to scanning the
#                      per-project transcript dir for the matching session_id.
#   - cwd              (working directory at session end)

set -u  # no -e: we must not abort on errors, fail-safe is paramount
input=$(cat 2>/dev/null || true)

# Best-effort jq; fall back to Python if jq missing.
extract() {
  local key=$1
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r ".${key} // empty" 2>/dev/null
  else
    printf '%s' "$input" | /usr/bin/python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('${key}', '') or '')
except Exception:
    print('')
" 2>/dev/null
  fi
}

session_id=$(extract "session_id")
transcript_path=$(extract "transcript_path")
cwd=$(extract "cwd")

# Fallback: locate transcript if the hook payload didn't include its path.
# Claude Code stores transcripts under ~/.claude/projects/{cwd-hash}/{uuid}.jsonl
if [[ -z "$transcript_path" && -n "$session_id" ]]; then
  projects_dir="$HOME/.claude/projects"
  if [[ -d "$projects_dir" ]]; then
    # Look for the most recent file whose basename starts with session_id.
    # ls -t (mtime newest first). Avoid `find` per the project's no-grep hook.
    for d in "$projects_dir"/*/; do
      candidate="${d}${session_id}.jsonl"
      if [[ -f "$candidate" ]]; then
        transcript_path="$candidate"
        break
      fi
    done
  fi
fi

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/capture-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  echo "---"
  echo "$ts session=$session_id cwd=$cwd transcript=$transcript_path"
} >> "$log" 2>/dev/null

# Skip if we couldn't find anything to capture.
if [[ -z "$transcript_path" || ! -f "$transcript_path" ]]; then
  echo "$ts skipped: no transcript found" >> "$log" 2>/dev/null
  exit 0
fi

# Locate the project's venv-installed `iai-mcp` CLI. Cache the last-known-good
# path in ~/.iai-mcp/.cli-path to avoid re-scanning on every session end.
cli_cache="$HOME/.iai-mcp/.cli-path"
iai_cli=""
if [[ -f "$cli_cache" ]]; then
  cached=$(cat "$cli_cache" 2>/dev/null || true)
  [[ -x "$cached" ]] && iai_cli="$cached"
fi
if [[ -z "$iai_cli" ]]; then
  # Resolve via PATH first (covers ~/.local/bin/iai-mcp installed by scripts/install.sh)
  path_cli="$(command -v iai-mcp 2>/dev/null || true)"
  if [[ -n "$path_cli" && -x "$path_cli" ]]; then
    iai_cli="$path_cli"
  else
    # Fall back to common clone locations
    for candidate in \
      "$HOME/.local/bin/iai-mcp" \
      "$HOME/iai-mcp/.venv/bin/iai-mcp" \
      "$HOME/IAI-MCP/.venv/bin/iai-mcp" \
      "/usr/local/bin/iai-mcp" \
      "/opt/homebrew/bin/iai-mcp"; do
      if [[ -x "$candidate" ]]; then
        iai_cli="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$iai_cli" ]]; then
    printf '%s' "$iai_cli" > "$cli_cache" 2>/dev/null || true
  fi
fi

if [[ -z "$iai_cli" ]]; then
  echo "$ts skipped: iai-mcp CLI not found" >> "$log" 2>/dev/null
  exit 0
fi

# Run capture with a 30s hard timeout — if it hangs, the session must still
# end cleanly. `timeout` is in coreutils (macOS: brew install coreutils). We
# fall back to a background kill loop if absent.
if command -v timeout >/dev/null 2>&1; then
  result=$(timeout 30 "$iai_cli" capture-transcript --no-spawn \
    --session-id "$session_id" \
    --max-turns 200 \
    "$transcript_path" 2>&1)
elif command -v gtimeout >/dev/null 2>&1; then
  result=$(gtimeout 30 "$iai_cli" capture-transcript --no-spawn \
    --session-id "$session_id" \
    --max-turns 200 \
    "$transcript_path" 2>&1)
else
  result=$("$iai_cli" capture-transcript --no-spawn \
    --session-id "$session_id" \
    --max-turns 200 \
    "$transcript_path" 2>&1)
fi
rc=$?

{
  echo "$ts rc=$rc result=$result"
} >> "$log" 2>/dev/null

exit 0
