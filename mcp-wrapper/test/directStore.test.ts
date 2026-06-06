// Tests for the direct-store fallthrough path.
//
// Cases (episodes_recent):
// 1. invokeTool - bridge.call() throws daemon-down error -> spawns direct recency CLI
// 2. handleToolCall - bridge.start() rejects -> routes to direct-store
// 3. handleToolCall - start() resolves but warm invokeTool error -> propagates
// 4. index.ts CallToolRequest handler (via buildServer factory) - start() rejects ->
// store-backed result (real MCP entry, no transport/daemon/home touch)
// Cases continued (memory_capture):
// 5. invokeTool memory_capture call-reject -> direct-write CLI
// 6. handleToolCall memory_capture start-reject -> direct-write CLI
// 7. index.ts CallToolRequest factory memory_capture start-reject
// Cases continued (memory_recall):
// 8. invokeTool memory_recall call-reject -> direct-recall CLI (direct-store primary)
// 9. handleToolCall memory_recall start-reject -> direct-recall CLI
// 10. index.ts CallToolRequest factory memory_recall start-reject

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { EventEmitter } from "node:events";

import { invokeTool, handleToolCall, runDirectRecency } from "../src/tools.js";
import type { PythonCoreBridge } from "../src/bridge.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type SpawnReturn = ReturnType<typeof import("node:child_process").spawn>;

/** Build a mock spawn function that emits a JSON payload + close 0. */
function makeMockSpawnFn(payload: Record<string, unknown>) {
  const calls: Array<{ cmd: string; args: string[] }> = [];
  const spawnFn = (cmd: string, args: ReadonlyArray<string>): SpawnReturn => {
    calls.push({ cmd, args: [...args] });
    const proc = new EventEmitter() as EventEmitter & {
      stdout: EventEmitter & { setEncoding: (enc: string) => void };
      stderr: EventEmitter;
      kill: () => void;
    };
    const stdout = new EventEmitter() as EventEmitter & {
      setEncoding: (enc: string) => void;
    };
    stdout.setEncoding = () => {};
    proc.stdout = stdout;
    proc.stderr = new EventEmitter();
    proc.kill = () => {};
    setImmediate(() => {
      stdout.emit("data", JSON.stringify(payload));
      proc.emit("close", 0);
    });
    return proc as unknown as SpawnReturn;
  };
  return { spawnFn, calls };
}

/** A mock spawn that records whether it was called (never emits data). */
function makeNeverCalledSpawnFn() {
  let wasCalled = false;
  const spawnFn = (cmd: string, args: ReadonlyArray<string>): SpawnReturn => {
    wasCalled = true;
    const proc = new EventEmitter() as EventEmitter & {
      stdout: EventEmitter & { setEncoding: (enc: string) => void };
      stderr: EventEmitter;
      kill: () => void;
    };
    const stdout = new EventEmitter() as EventEmitter & {
      setEncoding: (enc: string) => void;
    };
    stdout.setEncoding = () => {};
    proc.stdout = stdout;
    proc.stderr = new EventEmitter();
    proc.kill = () => {};
    // Never emits — caller times out or the test assertion fires first.
    return proc as unknown as SpawnReturn;
  };
  return { spawnFn, get wasCalled() { return wasCalled; } };
}

const DIRECT_RECENCY_PAYLOAD = {
  turns: [
    {
      record_id: "rec-1",
      literal_surface: "direct store turn",
      session_id: "test-session",
      captured_at: "2026-06-01T10:00:00+00:00",
    },
  ],
  count: 1,
};

// ---------------------------------------------------------------------------
// Case 1: invokeTool episodes_recent — bridge.call throws daemon-down error
// → falls through to direct-store CLI subcommand
// ---------------------------------------------------------------------------

describe("invokeTool episodes_recent daemon-down (call rejects)", () => {
  it("falls through to the direct-store CLI subcommand, NOT bank-recall", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECENCY_PAYLOAD);

    // Mock bridge whose call() throws a daemon-down error (socket dead).
    const mockBridge = {
      call: async () => { throw new Error("socket dead"); },
    } as unknown as PythonCoreBridge;

    const result = await invokeTool(mockBridge, "episodes_recent", { n: 5 }, spawnFn as any);

    // The spawned command must target the direct recency subcommand.
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai (not iai-mcp bank-recall)");
    assert.ok(
      args.includes("last"),
      `args must include 'last'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      args.includes("--json"),
      `args must include '--json'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      !args.includes("bank-recall"),
      "must NOT spawn bank-recall for recency",
    );

    // Result must be store-backed.
    assert.ok(result !== null, "result must be non-null");
    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
    assert.ok(Array.isArray(r["turns"]), "turns must be an array");
  });
});

// ---------------------------------------------------------------------------
// Case 2: handleToolCall episodes_recent — bridge.start() REJECTS
// → routes to direct store
// ---------------------------------------------------------------------------

describe("handleToolCall episodes_recent — bridge.start rejects", () => {
  it("routes to direct-store CLI when bridge.start() rejects with daemon-unreachable error", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECENCY_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ENOENT (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
    } as unknown as PythonCoreBridge;

    const result = await handleToolCall(mockBridge, "episodes_recent", { n: 5 }, spawnFn as any);

    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai");
    assert.ok(args.includes("last"), "args must include 'last'");
    assert.ok(args.includes("--json"), "args must include '--json'");
    assert.ok(!args.includes("bank-recall"), "must NOT spawn bank-recall");

    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});

// ---------------------------------------------------------------------------
// Case 3: handleToolCall — start() resolves but WARM invokeTool error
// → error PROPAGATES, no direct-store spawn
// ---------------------------------------------------------------------------

describe("handleToolCall — warm invokeTool error propagates", () => {
  it("rejects with the warm error and never spawns direct-store or bank-recall", async () => {
    const { spawnFn: neverSpawn, wasCalled } = makeNeverCalledSpawnFn();

    const warmError = new Error("RPC timeout: method episodes_recent timed out");

    const mockBridge = {
      start: async () => { /* resolves — daemon IS up */ },
      call: async () => { throw warmError; },
    } as unknown as PythonCoreBridge;

    await assert.rejects(
      () => handleToolCall(mockBridge, "episodes_recent", { n: 5 }, neverSpawn as any),
      (err: Error) => {
        assert.equal(
          err.message,
          warmError.message,
          "must propagate the original warm error",
        );
        return true;
      },
    );
    assert.equal(wasCalled, false, "direct-store spawn must NOT be triggered by a warm error");
  });
});

// ---------------------------------------------------------------------------
// Case 4: index.ts CallToolRequest handler (via buildServer factory) —
// bridge.start() REJECTS → store-backed result
//
// Imports the SIDE-EFFECT-FREE buildServer factory. No transport is spawned,
// no lifecycle code runs, no daemon/home is touched.
// ---------------------------------------------------------------------------

describe("index.ts buildServer CallToolRequest — start rejects → direct-store", () => {
  it("routes episodes_recent to direct-store via the factory-registered handler (mandatory real test)", async () => {
    // Import the factory — this must not trigger any daemon/home side effects.
    const { buildServer } = await import("../src/index.js");

    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECENCY_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ECONNREFUSED (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
      disconnect: () => {},
    } as unknown as PythonCoreBridge;

    const { server } = buildServer(mockBridge, spawnFn as any);

    // Reach the registered CallToolRequest handler via the server's internal
    // request-handler registry. The MCP SDK stores handlers in _requestHandlers
    // keyed by method; "tools/call" is the wire method for CallToolRequest.
    const requestHandlers = (server as any)._requestHandlers as Map<
      string,
      (req: unknown, extra: unknown) => Promise<unknown>
    >;
    const handler = requestHandlers.get("tools/call");
    assert.ok(handler, "CallToolRequest handler must be registered");

    const req = {
      method: "tools/call",
      params: {
        name: "episodes_recent",
        arguments: { n: 5 },
      },
    };
    const result = await handler(req, {}) as Record<string, unknown>;

    // The handler must return a store-backed result (not isError, not bank).
    assert.ok(!result["isError"], `must not be an error; got: ${JSON.stringify(result)}`);

    // structuredContent or content must carry the direct-store payload.
    const sc = (result["structuredContent"] ?? {}) as Record<string, unknown>;
    assert.equal(
      sc["_source"] ?? (JSON.parse((result["content"] as any)?.[0]?.text ?? "{}")["_source"]),
      "direct-store",
      "_source must be 'direct-store' in the response payload",
    );

    // Verify the subprocess was spawned with the right CLI args.
    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    assert.equal(calls[0].cmd, "iai");
    assert.ok(calls[0].args.includes("last"));
    assert.ok(calls[0].args.includes("--json"));
    assert.ok(!calls[0].args.includes("bank-recall"), "must not spawn bank-recall");
  });
});

// ---------------------------------------------------------------------------
// Case 5: memory_capture daemon-down (invokeTool, bridge.call rejects)
// → writes via the direct-store CLI subcommand (NOT bank)
// ---------------------------------------------------------------------------

const DIRECT_WRITE_PAYLOAD = {
  id: "rec-write-1",
  status: "inserted",
};

describe("invokeTool memory_capture daemon-down (call rejects) → direct-store write", () => {
  it("falls through to the direct-write CLI subcommand, NOT bank-recall", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_WRITE_PAYLOAD);

    // Mock bridge whose call() throws a daemon-down error (socket dead).
    const mockBridge = {
      call: async () => { throw new Error("socket dead"); },
    } as unknown as PythonCoreBridge;

    const result = await invokeTool(mockBridge, "memory_capture", { literal: "test capture text" }, spawnFn as any);

    // The spawned command must target the direct-write subcommand.
    assert.equal(calls.length, 1, "exactly one subprocess must be spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai (not iai-mcp bank-recall)");
    assert.ok(
      args.includes("capture"),
      `args must include 'capture'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      args.includes("--json"),
      `args must include '--json'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      !args.includes("bank-recall"),
      "must NOT spawn bank-recall for a write",
    );

    // Result must be store-backed.
    assert.ok(result !== null, "result must be non-null");
    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});

// ---------------------------------------------------------------------------
// Case 6: memory_capture daemon-DOWN via handleToolCall (bridge.start REJECTS)
// → writes direct
// ---------------------------------------------------------------------------

describe("handleToolCall memory_capture — bridge.start rejects → direct write", () => {
  it("routes to direct-write CLI when bridge.start() rejects with daemon-unreachable error", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_WRITE_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ENOENT (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
    } as unknown as PythonCoreBridge;

    const result = await handleToolCall(mockBridge, "memory_capture", { literal: "test text" }, spawnFn as any);

    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai");
    assert.ok(args.includes("capture"), "args must include 'capture'");
    assert.ok(args.includes("--json"), "args must include '--json'");
    assert.ok(!args.includes("bank-recall"), "must NOT spawn bank-recall");

    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});

// ---------------------------------------------------------------------------
// Case 7: index.ts CallToolRequest handler (via buildServer factory) —
// bridge.start() REJECTS for memory_capture → direct-store write
// ---------------------------------------------------------------------------

describe("index.ts buildServer CallToolRequest memory_capture — start rejects → direct-store write", () => {
  it("routes memory_capture to direct-write via the factory-registered handler (mandatory real test)", async () => {
    const { buildServer } = await import("../src/index.js");

    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_WRITE_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ECONNREFUSED (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
      disconnect: () => {},
    } as unknown as PythonCoreBridge;

    const { server } = buildServer(mockBridge, spawnFn as any);

    const requestHandlers = (server as any)._requestHandlers as Map<
      string,
      (req: unknown, extra: unknown) => Promise<unknown>
    >;
    const handler = requestHandlers.get("tools/call");
    assert.ok(handler, "CallToolRequest handler must be registered");

    const req = {
      method: "tools/call",
      params: {
        name: "memory_capture",
        arguments: { literal: "test capture text" },
      },
    };
    const result = await handler(req, {}) as Record<string, unknown>;

    // The handler must return a store-backed result (not isError, not bank).
    assert.ok(!result["isError"], `must not be an error; got: ${JSON.stringify(result)}`);

    // Verify the subprocess was spawned with the right CLI args.
    assert.equal(calls.length, 1, "exactly one subprocess spawned");
    assert.equal(calls[0].cmd, "iai");
    assert.ok(calls[0].args.includes("capture"), "must spawn capture subcommand");
    assert.ok(calls[0].args.includes("--json"));
    assert.ok(!calls[0].args.includes("bank-recall"), "must not spawn bank-recall");

    // Result must carry _source: "direct-store".
    const sc = (result["structuredContent"] ?? {}) as Record<string, unknown>;
    const source = sc["_source"] ?? (() => {
      try {
        return JSON.parse((result["content"] as any)?.[0]?.text ?? "{}")["_source"];
      } catch { return undefined; }
    })();
    assert.equal(source, "direct-store", "_source must be 'direct-store' in the response payload");
  });
});

// ---------------------------------------------------------------------------
// Case 8: memory_recall daemon-down (invokeTool, bridge.call rejects)
// → direct-recall CLI first (NOT bank)
// ---------------------------------------------------------------------------

const DIRECT_RECALL_PAYLOAD = {
  hits: [
    {
      literal_surface: "degraded store recall hit text",
      score: 0.0,
      _degraded: true,
    },
  ],
  count: 1,
  _source: "direct-store",
};

describe("invokeTool memory_recall daemon-down (call rejects) → direct-store recall", () => {
  it("falls through to the direct-recall CLI subcommand FIRST (not bank)", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const mockBridge = {
      call: async () => { throw new Error("socket dead"); },
    } as unknown as PythonCoreBridge;

    const result = await invokeTool(mockBridge, "memory_recall", { cue: "test recall cue", budget_tokens: 3000 }, spawnFn as any);

    // The FIRST spawned command must target the direct-recall subcommand.
    assert.ok(calls.length >= 1, "at least one subprocess must be spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai (not iai-mcp bank-recall)");
    assert.ok(
      args.includes("recall"),
      `first spawn args must include 'recall'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      args.includes("--json"),
      `first spawn args must include '--json'; got: ${JSON.stringify(args)}`,
    );
    assert.ok(
      !args.includes("bank-recall"),
      "must NOT spawn bank-recall as the primary daemon-down path",
    );

    // Result must be store-backed.
    assert.ok(result !== null, "result must be non-null");
    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});

// ---------------------------------------------------------------------------
// Case 9: memory_recall daemon-DOWN via handleToolCall (bridge.start REJECTS)
// → direct-recall CLI first
// ---------------------------------------------------------------------------

describe("handleToolCall memory_recall — bridge.start rejects → direct-store recall", () => {
  it("routes to direct-recall CLI when bridge.start() rejects", async () => {
    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ENOENT (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
    } as unknown as PythonCoreBridge;

    const result = await handleToolCall(mockBridge, "memory_recall", { cue: "test cue" }, spawnFn as any);

    assert.ok(calls.length >= 1, "at least one subprocess spawned");
    const { cmd, args } = calls[0];
    assert.equal(cmd, "iai", "must spawn iai");
    assert.ok(args.includes("recall"), "first spawn must include 'recall'");
    assert.ok(args.includes("--json"), "first spawn must include '--json'");
    assert.ok(!args.includes("bank-recall"), "must NOT spawn bank-recall as primary");

    const r = result as Record<string, unknown>;
    assert.equal(r["_source"], "direct-store", "_source must be 'direct-store'");
  });
});

// ---------------------------------------------------------------------------
// Case 10: index.ts CallToolRequest handler — bridge.start() REJECTS for
// memory_recall → direct-store (real MCP entry)
// ---------------------------------------------------------------------------

describe("index.ts buildServer CallToolRequest memory_recall — start rejects → direct-store", () => {
  it("routes memory_recall to direct-store via the factory-registered handler (mandatory real test)", async () => {
    const { buildServer } = await import("../src/index.js");

    const { spawnFn, calls } = makeMockSpawnFn(DIRECT_RECALL_PAYLOAD);

    const mockBridge = {
      start: async () => {
        const err = new Error("daemon_unreachable: socket ECONNREFUSED (code -32002)");
        (err as any).name = "DaemonUnreachableError";
        throw err;
      },
      call: async () => { throw new Error("never reached"); },
      disconnect: () => {},
    } as unknown as PythonCoreBridge;

    const { server } = buildServer(mockBridge, spawnFn as any);

    const requestHandlers = (server as any)._requestHandlers as Map<
      string,
      (req: unknown, extra: unknown) => Promise<unknown>
    >;
    const handler = requestHandlers.get("tools/call");
    assert.ok(handler, "CallToolRequest handler must be registered");

    const req = {
      method: "tools/call",
      params: {
        name: "memory_recall",
        arguments: { cue: "test recall cue", budget_tokens: 3000 },
      },
    };
    const result = await handler(req, {}) as Record<string, unknown>;

    // The handler must return a store-backed result (not isError, not bank).
    assert.ok(!result["isError"], `must not be an error; got: ${JSON.stringify(result)}`);

    // Verify the FIRST spawned subprocess was the direct-recall CLI.
    assert.ok(calls.length >= 1, "at least one subprocess spawned");
    assert.equal(calls[0].cmd, "iai");
    assert.ok(calls[0].args.includes("recall"), "first spawn must include 'recall' subcommand");
    assert.ok(calls[0].args.includes("--json"));
    assert.ok(!calls[0].args.includes("bank-recall"), "must not spawn bank-recall as primary");

    // Result must carry _source: "direct-store".
    const sc = (result["structuredContent"] ?? {}) as Record<string, unknown>;
    const source = sc["_source"] ?? (() => {
      try {
        return JSON.parse((result["content"] as any)?.[0]?.text ?? "{}")["_source"];
      } catch { return undefined; }
    })();
    assert.equal(source, "direct-store", "_source must be 'direct-store' in the response payload");
  });
});
