#!/usr/bin/env node
// IAI-MCP TypeScript wrapper entry point (Plan 03 wave).
//
// - Spawns the Python core over stdio JSON-RPC (see bridge.ts)
// - Advertises the 12 hot tools via HOT_TOOLS registry (TOK-02)
// - Attaches Anthropic 1h-TTL cache_control at the stable/volatile boundary
//   (TOK-01) via caching.ts helpers
// - Advertises `clear_tool_uses_20250919` context editing with 30k trigger
//   (TOK-05) via registry.ts CONTEXT_EDITING_CONFIG
// - On MCP `initialize`, warms the Python session_start payload so the first
//   real user turn doesn't pay the fresh-session cost synchronously.

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import {
  emitSessionOpen,
  newSessionId,
  PythonCoreBridge,
} from "./bridge.js";
import {
  applyCacheBreakpoint,
  buildCachedSystemPrompt,
  type ContentBlock,
  type SessionPayloadRaw,
} from "./caching.js";
import { WrapperLifecycle } from "./lifecycle.js";
import {
  CONTEXT_EDITING_CONFIG,
  HOT_TOOLS,
  listHotTools,
} from "./registry.js";
import { invokeTool, type ToolName } from "./tools.js";

// Re-export so consumers of the module (and tests) can touch the helpers
// without dynamic imports.
export {
  applyCacheBreakpoint,
  buildCachedSystemPrompt,
  CONTEXT_EDITING_CONFIG,
  HOT_TOOLS,
};
export type { ContentBlock, SessionPayloadRaw };

// ---------------------------------------------------------------------------
// mcp-tools-list-empty-cache fix (2026-05-02):
//
// Pre-fix order was:
//   1. await bridge.start()           ← could block 5s on slow daemon
//   2. construct Server + handlers
//   3. await server.connect(transport)
//
// On a slow daemon (cold launchd hand-off, multi-second LanceDB open, RSS
// watchdog respawn) the top-level await in step 1 delayed step 3 past the
// MCP client's tools/list timeout. The client cached an empty tool list
// for the rest of the session — symptom: "Connected" but zero
// `mcp__iai-mcp__*` tools in the registry.
//
// Fixed order is:
//   1. construct Server + register both request handlers + assign
//      oninitialized (must be set before connect — the initialized
//      notification fires immediately after handshake and an unset
//      handler would discard the HIPPEA pre-warm trigger).
//   2. await server.connect(transport)  ← tools/list is responsive HERE,
//      independent of daemon state (handler returns from static
//      registry.listHotTools()).
//   3. fire-and-forget bridge.start() chained with emitSessionOpen — the
//      D5-05 invariant "emitSessionOpen fires AFTER daemon socket
//      reachable" is preserved by the .then() chain.
//   4. CallToolRequest handler lazy-awaits bridge.start() before
//      delegating to invokeTool — first tools/call may pay daemon
//      cold-start cost ONCE; tools/list never blocks.
//
// Invariants preserved:
//   - Phase 7.1: wrapper does NOT spawn daemon (bridge.ts unchanged on
//     this point — it's still socket-only).
//   - Plan 05-04 D5-05 (HIPPEA pre-warm): emitSessionOpen still chained
//     off bridge.start() readiness.
//   - Plan 07-04 Task 2: SIGTERM/SIGINT closes socket only; daemon
//     survives. Unchanged.
// ---------------------------------------------------------------------------

const bridge = new PythonCoreBridge();

const server = new Server(
  {
    name: "iai-mcp",
    version: "0.1.0",
  },
  {
    capabilities: { tools: {} },
    // Expose TOK-05 context-editing config so MCP hosts that honour
    // Anthropic's context management can pick it up at discovery time.
    instructions: JSON.stringify({
      context_editing: CONTEXT_EDITING_CONFIG,
      hot_tools: HOT_TOOLS,
    }),
  },
);

// tools/list MUST return from the static registry without touching the
// bridge — see file-top comment block. This is what makes the wrapper
// safe to advertise to the MCP client before the daemon socket is
// reachable.
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: listHotTools(),
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const name = req.params.name as ToolName;
  if (!HOT_TOOLS.includes(name)) {
    return {
      content: [{ type: "text" as const, text: `unknown tool ${name}` }],
      isError: true,
    };
  }
  try {
    // Lazy bridge connect: the first tools/call after wrapper boot drives
    // the daemon socket connect. Subsequent calls short-circuit on the
    // alive socket. start() is concurrency-safe (startPromise serialises
    // multiple concurrent first-callers — see bridge.ts).
    await bridge.start();
    const result = await invokeTool(bridge, name, req.params.arguments ?? {});
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result) }],
    };
  } catch (e) {
    return {
      content: [
        { type: "text" as const, text: `error: ${(e as Error).message}` },
      ],
      isError: true,
    };
  }
});

// Boot-time session id for Plan 05-04 session_open + downstream bookkeeping.
const bootSessionId = newSessionId();

// MCP initialize hook -- warm the Python session-start payload so the first
// real turn doesn't pay the fresh-session cost synchronously. OPS-05 continuity
// is surfaced earlier this way: by the time Claude issues tools/call, the L0
// pinned record is already resident in the Python core's warm cache.
//
// Must be assigned BEFORE server.connect() — the initialized notification
// fires immediately after the handshake and an unset handler would silently
// discard the pre-warm trigger.
server.oninitialized = () => {
  // Chain on bridge readiness so the session_start_payload call doesn't
  // race the socket connect. start() is idempotent and serialised; if
  // the lazy CallToolRequest path already drove start, this awaits the
  // same in-flight promise.
  bridge
    .start()
    .then(() =>
      bridge.call<SessionPayloadRaw>("session_start_payload", {
        session_id: bootSessionId,
      }),
    )
    .catch(() => null);
};

// Phase 10.5 L5 + L4: proactive wake + heartbeat refresh.
//
// Run BEFORE server.connect so the heartbeat is registered before any
// tools/list or tools/call request can land. ensureDaemonAlive is
// independent of the bridge.start() call below — it only probes the
// socket and (on darwin) invokes `launchctl kickstart` via execFile;
// it never connects. The 045999b decoupling is preserved: tools/list
// still responds from the static registry whether the daemon is up
// or not, and ensureDaemonAlive's failure path (wake.signal write)
// is silent and non-fatal.
const lifecycle = new WrapperLifecycle();
await lifecycle.ensureDaemonAlive();
await lifecycle.registerHeartbeat();

const transport = new StdioServerTransport();
await server.connect(transport);

// Fire-and-forget daemon connect AFTER the MCP transport is live.
// - bridge.start(): socket-only connect to the singleton daemon (Phase 7.1
//   invariant — never spawns).
// - emitSessionOpen: D5-05 HIPPEA pre-warm hint; chained off start() so
//   the cascade-LRU activation happens AFTER the daemon is known
//   reachable. If the daemon is unreachable, start() rejects with
//   DaemonUnreachableError and the .catch() suppresses the unhandled
//   rejection — the wrapper continues serving tools/list and falls back
//   to per-call lazy retry in the CallToolRequest handler.
void bridge
  .start()
  .then(() => emitSessionOpen(bootSessionId))
  .catch(() => {
    // Silent: tools/call will surface the daemon_unreachable error
    // synchronously when the user actually invokes a tool.
  });

// Phase 7 (Plan 07-04 Task 2): wrapper closing must NOT kill the shared
// daemon. disconnect() closes the socket only; the singleton survives so
// other wrappers (other MCP hosts, sub-agents) and future boots
// can join. This is the load-bearing semantic of the Phase 7 singleton
// model — the pre-Phase-7 wrapper-side child-kill API has been removed.
//
// Phase 10.5 L4 addition: cleanupHeartbeat clears the refresh timer
// AND deletes ~/.iai-mcp/wrappers/heartbeat-<pid>-<uuid>.json so the
// daemon-side scanner doesn't have to rely on STALE-detection for a
// gracefully-exiting wrapper. Cleanup is idempotent and never throws.
const shutdown = async (): Promise<void> => {
  try {
    await lifecycle.cleanupHeartbeat();
  } catch {
    // Cleanup is best-effort; the daemon's HeartbeatScanner reaps
    // STALE / ORPHAN entries on its next tick.
  }
  bridge.disconnect();
  process.exit(0);
};
process.on("SIGTERM", () => {
  void shutdown();
});
process.on("SIGINT", () => {
  void shutdown();
});
