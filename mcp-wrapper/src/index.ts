#!/usr/bin/env node
// IAI-MCP TypeScript wrapper entry point.
//
// - Spawns the Python core over stdio JSON-RPC (see bridge.ts)
// - Advertises the 12 hot tools via HOT_TOOLS registry
// - Attaches Anthropic 1h-TTL cache_control at the stable/volatile boundary
//   via caching.ts helpers
// - Advertises `clear_tool_uses_20250919` context editing with 30k trigger
//   via registry.ts CONTEXT_EDITING_CONFIG
// - On MCP `initialize`, warms the Python session_start payload so the first
// real user turn doesn't pay the fresh-session cost synchronously.
//
// ALL startup side effects are isolated inside the guarded main() function
// (import.meta.url entrypoint check). Importing this module or calling
// buildServer() touches NO daemon and NO home directory — making the module
// safe to import in tests for a real CallToolRequest.

import { pathToFileURL } from "node:url";
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
import {
  emitSickWarningIfNeeded,
  probeDaemonDoctor,
} from "./sickWarning.js";
import { spawn } from "node:child_process";
import { handleToolCall, type ToolName } from "./tools.js";

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
// mcp-tools-list-empty-cache fix:
//
// Fixed order (preserved after CL4-H2 refactor):
// 1. buildServer(): construct Server + register both request handlers +
// assign oninitialized (must be set before connect).
// 2. main(): await lifecycle + await server.connect(transport).
// 3. main(): fire-and-forget bridge.start() chained with emitSessionOpen.
// 4. CallToolRequest handler delegates to handleToolCall (lazy bridge start
// inside handleToolCall, first call may pay daemon cold-start cost).
//
// Invariants preserved:
// -: wrapper does NOT spawn daemon (bridge.ts unchanged).
// - (pre-warm): emitSessionOpen still chained off
// bridge.start() readiness.
// - SIGTERM/SIGINT closes socket only; daemon survives.
// ---------------------------------------------------------------------------

// MCP spec 2025-03-26+ requires a `structuredContent` field on a tool-call
// response whenever the tool declares an `outputSchema`; spec-compliant
// clients reject a schema-bearing tool's response with JSON-RPC -32600 if it
// is absent. Build the response so the text content is always present and
// `structuredContent` is added only for object-shaped payloads (the spec
// permits structuredContent only as a JSON object — exactly what the output
// schemas describe). Non-object payloads return the content-only shape.
function toolResult(payload: unknown) {
  const content = [
    { type: "text" as const, text: JSON.stringify(payload) },
  ];
  if (typeof payload === "object" && payload !== null) {
    return {
      content,
      structuredContent: payload as Record<string, unknown>,
    };
  }
  return { content };
}

// ---------------------------------------------------------------------------
// buildServer: SIDE-EFFECT-FREE factory (CL4-H2).
//
// Constructs the Server, registers ListTools + CallTool handlers, and assigns
// the oninitialized prewarm callback (no side effect until a transport
// connects, so binding here is safe).
// Does NOT connect, does NOT call lifecycle, does NOT call bridge.start —
// importing this module and calling buildServer() touches NO daemon/home.
// ---------------------------------------------------------------------------
export function buildServer(
  bridge?: PythonCoreBridge,
  spawnFn: typeof spawn = spawn,
): { server: Server; bridge: PythonCoreBridge } {
  const b = bridge ?? new PythonCoreBridge();

  const server = new Server(
    {
      name: "iai-mcp",
      version: "1.0.0",
    },
    {
      capabilities: { tools: {} },
      // Expose context-editing config so MCP hosts that honour
      // Anthropic's context management can pick it up at discovery time.
      instructions: JSON.stringify({
        context_editing: CONTEXT_EDITING_CONFIG,
        hot_tools: HOT_TOOLS,
      }),
    },
  );

  // tools/list MUST return from the static registry without touching the
  // bridge — see file-top comment block. This is what makes the wrapper
  // safe to advertise to the MCP client before the daemon socket is reachable.
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: listHotTools(),
  }));

  // CallToolRequest handler: TRIVIAL delegate to the shared handleToolCall.
  // All per-tool routing (incl. daemon-down fallbacks) lives in handleToolCall
  // in tools.ts — no per-tool branch here.
  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const name = req.params.name as ToolName;
    if (!HOT_TOOLS.includes(name)) {
      return {
        content: [{ type: "text" as const, text: `unknown tool ${name}` }],
        isError: true,
      };
    }
    try {
      const result = await handleToolCall(b, name, req.params.arguments ?? {}, spawnFn);
      return toolResult(result);
    } catch (e) {
      return {
        content: [
          { type: "text" as const, text: `error: ${(e as Error).message}` },
        ],
        isError: true,
      };
    }
  });

  // Bind oninitialized: no effect until a transport connects (safe in factory).
  // Production main() connects; the prewarm only fires in the real entrypoint.
  const bootSessionId = newSessionId();
  server.oninitialized = () => {
    // Chain on bridge readiness so the session_start_payload call doesn't
    // race the socket connect. start() is idempotent and serialised; if
    // the lazy CallToolRequest path already drove start, this awaits the
    // same in-flight promise.
    b.start()
      .then(() =>
        b.call<SessionPayloadRaw>("session_start_payload", {
          session_id: bootSessionId,
        }),
      )
      .catch(() => null);

    // Parallel fire-and-forget chain: probe the top-level `iai-mcp doctor`
    // CLI and write one stderr line if it reports FAIL.
    void probeDaemonDoctor()
      .then(emitSickWarningIfNeeded)
      .catch(() => null);
  };

  return { server, bridge: b };
}

// ---------------------------------------------------------------------------
// main(): ALL startup side effects live here — guarded by the entrypoint
// check so they ONLY run when the module is the process entry point.
// (1) lifecycle.ensureDaemonAlive() + lifecycle.registerHeartbeat()
// (2) server.connect(transport)
// (3) fire-and-forget bridge.start() + emitSessionOpen
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  const { server, bridge: b } = buildServer();

  // L5 + L4: proactive wake + heartbeat refresh.
  // Run BEFORE server.connect so the heartbeat is registered before any
  // tools/list or tools/call request can land.
  const lifecycle = new WrapperLifecycle();
  await lifecycle.ensureDaemonAlive();
  await lifecycle.registerHeartbeat();

  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Fire-and-forget daemon connect AFTER the MCP transport is live.
  void b
    .start()
    .then(() => emitSessionOpen(newSessionId()))
    .catch(() => {
      // Silent: tools/call will surface the daemon_unreachable error
      // synchronously when the user actually invokes a tool.
    });

  //: wrapper closing must NOT kill the shared daemon.
  // disconnect() closes the socket only; the singleton survives.
  // L4: cleanupHeartbeat clears the timer AND removes the
  // heartbeat file so the daemon scanner doesn't rely on STALE-detection.
  const shutdown = async (): Promise<void> => {
    try {
      await lifecycle.cleanupHeartbeat();
    } catch {
      // Cleanup is best-effort; the daemon's HeartbeatScanner reaps
      // STALE / ORPHAN entries on its next tick.
    }
    b.disconnect();
    process.exit(0);
  };
  process.on("SIGTERM", () => { void shutdown(); });
  process.on("SIGINT", () => { void shutdown(); });
}

// CL4-H2: entrypoint guard — ALL four startup side effects run ONLY here.
// Null-check process.argv[1] guards against environments where it may be
// undefined (e.g. Node REPL). When running as a test import this block
// is skipped entirely so NO daemon/home is touched.
if (
  process.argv[1] != null &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  void main();
}
