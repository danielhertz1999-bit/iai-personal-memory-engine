// Smoke for the bank-recall fallback path on socket-dead memory_recall.

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { EventEmitter } from "node:events";

import { runBankFallback, invokeTool } from "../src/tools.js";
import type { PythonCoreBridge } from "../src/bridge.js";

describe("runBankFallback", () => {
  it("spawns iai-mcp bank-recall and tags the response with _source", async () => {
    const calls: Array<{ cmd: string; args: string[] }> = [];
    const mockSpawn = (cmd: string, args: ReadonlyArray<string>) => {
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
      // Emit payload + close on the next tick so the await chain has a
      // chance to attach its listeners first.
      setImmediate(() => {
        stdout.emit(
          "data",
          JSON.stringify({
            hits: [
              {
                record_id: "abc",
                score: 0.5,
                reason: "bank-substring-match (processed)",
                literal_surface: "hello carrot",
                adjacent_suggestions: [],
                valid_from: null,
                valid_to: null,
              },
            ],
            anti_hits: [],
            activation_trace: [],
            budget_used: 0,
            cue_mode: "verbatim",
            patterns_observed: [],
            _knobs_applied: {},
          }),
        );
        proc.emit("close", 0);
      });
      return proc as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    };
    const result = await runBankFallback("carrot", 20, mockSpawn as any);
    assert.ok(result, "expected non-null fallback payload");
    assert.equal(result!["_source"], "bank-fallback");
    assert.equal((result!["hits"] as any[]).length, 1);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].cmd, "iai-mcp");
    assert.deepEqual(calls[0].args, [
      "bank-recall",
      "--query",
      "carrot",
      "--limit",
      "20",
      "--json",
    ]);
  });

  it("returns null when subprocess exits non-zero", async () => {
    const mockSpawn = () => {
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
      setImmediate(() => proc.emit("close", 1));
      return proc as unknown as ReturnType<
        typeof import("node:child_process").spawn
      >;
    };
    const result = await runBankFallback("anything", 5, mockSpawn as any);
    assert.equal(result, null);
  });
});

// Wiring-layer regression guard: when BOTH the bridge call AND the direct-recall
// subcommand fail, invokeTool must pass BANK_FALLBACK_LIMIT as --limit to the
// bank-recall subprocess regardless of what the caller supplies in args.budget_tokens.
//
// Updated for the new fallthrough chain (direct-store FIRST, bank LAST):
// the mockSpawnFn makes the direct-recall spawn fail (exit 1) so the
// budget-vs-limit guard is still valid for the LAST-RESORT bank path.
describe("invokeTool memory_recall budget-vs-limit decoupling", () => {
  it(
    "invokeTool budget-vs-limit decoupling: when direct-store + bridge both fail, bank receives limit=BANK_FALLBACK_LIMIT (20), NOT budget_tokens",
    async () => {
      const spawnCalls: Array<{ cmd: string; args: string[] }> = [];
      let spawnCount = 0;

      const mockSpawnFn = (cmd: string, args: ReadonlyArray<string>) => {
        const callIndex = spawnCount++;
        spawnCalls.push({ cmd, args: [...args] });

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
          if (callIndex === 0) {
            // First spawn: direct-recall CLI — make it fail (exit 1)
            // so the fallback chain continues to bank.
            proc.emit("close", 1);
          } else {
            // Second spawn: bank-recall — return a valid payload.
            stdout.emit(
              "data",
              JSON.stringify({
                hits: [],
                anti_hits: [],
                activation_trace: [],
                budget_used: 0,
                cue_mode: "verbatim",
                patterns_observed: [],
                _knobs_applied: {},
              }),
            );
            proc.emit("close", 0);
          }
        });
        return proc as unknown as ReturnType<
          typeof import("node:child_process").spawn
        >;
      };

      const mockBridge = {
        call: async () => { throw new Error("socket dead"); },
      } as unknown as PythonCoreBridge;

      // Ensure the fallback branch is not disabled in this test environment.
      const prevFallback = process.env["IAI_MCP_BANK_FALLBACK"];
      delete process.env["IAI_MCP_BANK_FALLBACK"];
      try {
        await invokeTool(
          mockBridge,
          "memory_recall",
          { cue: "test", budget_tokens: 9999 },
          mockSpawnFn as any,
        );
      } finally {
        if (prevFallback !== undefined) {
          process.env["IAI_MCP_BANK_FALLBACK"] = prevFallback;
        }
      }

      // Two spawns expected: [0] direct-recall (fails), [1] bank-recall.
      assert.ok(spawnCalls.length >= 2, `expected at least 2 spawns; got ${spawnCalls.length}`);

      // The bank-recall spawn must have --limit BANK_FALLBACK_LIMIT (20), NOT 9999.
      const bankArgs = spawnCalls[1].args;
      const limitIdx = bankArgs.indexOf("--limit");
      assert.ok(limitIdx >= 0, "--limit flag must be present in bank-recall argv");
      assert.equal(
        bankArgs[limitIdx + 1],
        "20",
        "invokeTool must pass BANK_FALLBACK_LIMIT (20) as --limit to bank, not budget_tokens",
      );
      assert.notEqual(
        bankArgs[limitIdx + 1],
        "9999",
        "--limit must not equal budget_tokens (9999)",
      );
    },
  );
});
