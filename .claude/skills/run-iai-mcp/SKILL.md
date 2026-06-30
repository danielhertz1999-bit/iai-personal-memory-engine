---
name: run-iai-mcp
description: Build, launch, and drive the iai-mcp personal memory engine (the local MCP daemon). Use when asked to run, start, smoke-test, screenshot, or verify the iai-mcp daemon, embedder, or capture/recall flow, or to check that the memory engine works after a change.
---

# Run iai-mcp (personal memory engine)

`iai-mcp` is a **background daemon + CLI**, not a GUI or web app. The daemon
(`python -m iai_mcp.daemon`) holds the SQLite memory store and the native Rust
embedder; the `iai` / `iai-mcp` CLIs talk to it over a Unix socket. You drive it
with the committed harness **`.claude/skills/run-iai-mcp/driver.sh`**, which
launches the daemon and exercises the real user surface (status, capture,
recall, doctor) and asserts the native Rust embedder is live.

All paths below are relative to the repo root (`<unit>/`).

## Run (agent path) — use the driver

```bash
bash .claude/skills/run-iai-mcp/driver.sh smoke
```

Exit 0 means: daemon came UP, `iai capture` was accepted, `iai recall` returned
ranked rows, and doctor reports `(v) native Rust embedder … backend=rust,
384-dim`. On success the last line is `==> SMOKE PASS`.

Sub-commands (all verified):

```bash
bash .claude/skills/run-iai-mcp/driver.sh up       # start daemon, wait until UP
bash .claude/skills/run-iai-mcp/driver.sh status   # one status round-trip
bash .claude/skills/run-iai-mcp/driver.sh down      # SIGTERM the daemon
```

The driver exports `IAI_MCP_EMBED_OFFLINE=1` for you (see Prerequisites) and
writes daemon stdout/stderr to `~/.iai-mcp/logs/skill-daemon.out`.

## Prerequisites

The CLI is installed as `iai` / `iai-mcp` (editable install of this repo, built
via `scripts/install.sh` — Rust engine via setuptools-rust, plus the
`mcp-wrapper` npm build). Confirm it's reachable:

```bash
which iai iai-mcp
```

The native Rust embedder needs the `bge-small-en-v1.5` model. In a restricted
container there's no HuggingFace network, so the model must already be in the
local HF cache. Verify it's there (the driver and daemon both rely on this):

```bash
ls ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*/
# expect: config.json  model.safetensors  tokenizer.json
```

If it's missing, fetch it once (needs network) before running the daemon:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5')"
```

With the cache present, `IAI_MCP_EMBED_OFFLINE=1` makes the embedder use it and
skip the network. The daemon auto-detects the cache too, but the driver sets the
var explicitly for determinism.

## Run (manual, without the driver)

Start the daemon in the background and round-trip the CLI:

```bash
IAI_MCP_EMBED_OFFLINE=1 nohup python -m iai_mcp.daemon > ~/.iai-mcp/logs/skill-daemon.out 2>&1 &
sleep 12
iai status                     # -> daemon UP, records N
iai capture 'hello from a manual run'
iai recall 'hello'             # -> ranked rows
iai-mcp doctor                 # 25-point health check
```

## Test (sanity, not the main event)

```bash
python -m pytest tests/ -q -m "not slow"
```

## Gotchas

- **There is no `iai-mcp daemon run`.** The daemon backend is the module
  `python -m iai_mcp.daemon`. The `iai-mcp daemon` subcommands (`install`,
  `start`, `stop`, …) are OS-service wrappers (launchd/systemd/Task Scheduler),
  not a foreground runner. In a bare container you launch the module directly.
- **`doctor` exits 1 on the `(e)` daemon-state check** with
  `fsm_state='TRANSITIONING'`. This is harmless in a freshly-launched container
  daemon — the FSM is mid-transition — and does NOT mean the engine is broken.
  The load-bearing check is `(v) native Rust embedder`. Do not gate success on
  doctor's exit code; grep for the `(v)` line instead (the driver does this).
- **Never pipe `iai-mcp doctor` directly under `set -o pipefail`.** Its exit 1
  poisons the whole pipeline even when your grep matches. Redirect to a file
  first (`doctor > out 2>&1 || true`), then grep the file.
- **`iai recall` prints `(daemon unreachable — store recall)` yet still works.**
  The sync recall client falls back to reading the store directly when it can't
  reuse the daemon socket; results are still ranked and correct. Not an error.
- **`(o) Claude subscription credentials … not_subscription` WARN is expected**
  in a container with no `claude /login`. The daemon falls back to local Tier-0
  consolidation. Not a failure.

## Troubleshooting

- **`iai status` shows `daemon DOWN` after launch** → tail the log:
  `tail -20 ~/.iai-mcp/logs/skill-daemon.out`. Most common cause is a second
  daemon already holding the exclusive store lock (only one runs at a time) —
  `pkill -f 'iai_mcp\.daemon'`, wait 2s, relaunch.
- **`doctor (v)` reports a non-rust backend or an embedder error** → the HF
  cache is missing or `IAI_MCP_EMBED_OFFLINE` isn't set. Re-check Prerequisites.
- **`daemon did not come UP within 20s`** → the Rust extension may be unbuilt;
  rebuild with `iai-mcp build-native` (or re-run `pip install -e .`).
