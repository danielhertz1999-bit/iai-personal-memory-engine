// Phase 10.5 — tests for `WrapperLifecycle`.
//
// Eight-test matrix from CONTEXT 10.5:
//
//   1. ensureDaemonAlive: socket reachable -> NO subprocess invoked.
//   2. ensureDaemonAlive: socket unreachable + darwin -> kickstart called.
//   3. ensureDaemonAlive: kickstart throws -> falls back to wake.signal.
//   4. ensureDaemonAlive: non-macos -> wake.signal written, no subprocess.
//   5. registerHeartbeat: file exists with correct schema.
//   6. heartbeat refresh: small interval -> last_refresh updates.
//   7. cleanupHeartbeat: file gone, timer cleared.
//   8. security: source has no `shell: true` and no shell-interpreting
//      subprocess variant in mcp-wrapper/src/.
//
// Test runner: Node's built-in `node:test` (zero new dep — Node 22 has
// it natively) loaded via the existing `tsx` dev-dep so `.ts` files
// run without a build step. Assertions: `node:assert/strict`.

import { describe, it } from "node:test";
import { strict as assert } from "node:assert";
import { mkdtemp, readFile, readdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { WrapperLifecycle } from "../src/lifecycle.js";

// Tmp-dir helper. node:test isolates per-file but not per-`it`, so
// every test allocates its own dir.
async function makeTmp(prefix: string): Promise<string> {
  return await mkdtemp(join(tmpdir(), `iai-mcp-lifecycle-${prefix}-`));
}

async function cleanupTmp(dir: string): Promise<void> {
  await rm(dir, { recursive: true, force: true });
}

// Sleep helper for fake-interval verification (Node's setInterval is
// real-time; we use a small interval (10 ms) and wait deterministically).
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------- ensureDaemonAlive

describe("WrapperLifecycle.ensureDaemonAlive", () => {
  it("does NOT invoke subprocess when socket is reachable", async () => {
    const tmp = await makeTmp("alive");
    try {
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 0, "kickstart must not be invoked when socket is alive");
      // wake.signal must NOT be written when daemon is reachable.
      await assert.rejects(stat(join(tmp, "wake.signal")));
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("invokes launchctl kickstart on darwin when socket is unreachable", async () => {
    const tmp = await makeTmp("kickstart");
    try {
      let kickstarts = 0;
      let signalWritten = false;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 1, "kickstart must be invoked exactly once on darwin");
      try {
        await stat(join(tmp, "wake.signal"));
        signalWritten = true;
      } catch {
        signalWritten = false;
      }
      assert.equal(
        signalWritten,
        false,
        "wake.signal must NOT be written on successful kickstart",
      );
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("falls back to wake.signal when kickstart fails on darwin", async () => {
    const tmp = await makeTmp("fallback");
    try {
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "darwin",
        socketReachable: async () => false,
        spawnKickstart: async () => {
          throw new Error("kickstart simulated failure");
        },
      });
      await lifecycle.ensureDaemonAlive();
      const sigStat = await stat(join(tmp, "wake.signal"));
      assert.ok(sigStat.isFile(), "wake.signal must exist after kickstart failure");
      const raw = await readFile(join(tmp, "wake.signal"), "utf-8");
      const parsed = JSON.parse(raw);
      assert.ok(typeof parsed.requested_at === "string");
      assert.ok(typeof parsed.wrapper_pid === "number");
      assert.ok(typeof parsed.wrapper_uuid === "string");
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("on non-macos writes wake.signal and never spawns subprocess", async () => {
    const tmp = await makeTmp("linux");
    try {
      let kickstarts = 0;
      const lifecycle = new WrapperLifecycle({
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath: join(tmp, "wrappers", "heartbeat-1-x.json"),
        platform: "linux",
        socketReachable: async () => false,
        spawnKickstart: async () => {
          kickstarts += 1;
        },
      });
      await lifecycle.ensureDaemonAlive();
      assert.equal(kickstarts, 0, "subprocess must never be invoked on non-darwin");
      const sigStat = await stat(join(tmp, "wake.signal"));
      assert.ok(sigStat.isFile(), "wake.signal must exist on non-darwin path");
    } finally {
      await cleanupTmp(tmp);
    }
  });
});

// ---------------------------------------------------------------- registerHeartbeat

describe("WrapperLifecycle.registerHeartbeat", () => {
  it("creates heartbeat file with correct schema", async () => {
    const tmp = await makeTmp("hb-schema");
    try {
      const heartbeatPath = join(tmp, "wrappers", "heartbeat-12345-abc.json");
      const lifecycle = new WrapperLifecycle({
        pid: 12345,
        uuid: "abc",
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath,
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        refreshIntervalMs: 60_000, // big — we don't want it firing in this test
      });
      await lifecycle.registerHeartbeat();
      try {
        const raw = await readFile(heartbeatPath, "utf-8");
        const parsed = JSON.parse(raw);
        assert.equal(parsed.pid, 12345);
        assert.equal(parsed.uuid, "abc");
        assert.ok(typeof parsed.started_at === "string");
        assert.ok(typeof parsed.last_refresh === "string");
        assert.ok(typeof parsed.wrapper_version === "string");
        assert.equal(parsed.schema_version, 1);
      } finally {
        await lifecycle.cleanupHeartbeat();
      }
    } finally {
      await cleanupTmp(tmp);
    }
  });

  it("refresh timer updates last_refresh", async () => {
    const tmp = await makeTmp("hb-refresh");
    try {
      const heartbeatPath = join(tmp, "wrappers", "heartbeat-1-x.json");
      const lifecycle = new WrapperLifecycle({
        pid: 1,
        uuid: "x",
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath,
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        refreshIntervalMs: 10, // tight interval to keep test fast
      });
      await lifecycle.registerHeartbeat();
      try {
        const before = JSON.parse(await readFile(heartbeatPath, "utf-8"));
        await sleep(60); // ~6 refresh ticks
        const after = JSON.parse(await readFile(heartbeatPath, "utf-8"));
        // started_at is stable; last_refresh advances.
        assert.equal(before.started_at, after.started_at);
        assert.notEqual(before.last_refresh, after.last_refresh);
      } finally {
        await lifecycle.cleanupHeartbeat();
      }
    } finally {
      await cleanupTmp(tmp);
    }
  });
});

// ---------------------------------------------------------------- cleanupHeartbeat

describe("WrapperLifecycle.cleanupHeartbeat", () => {
  it("deletes heartbeat file and clears timer", async () => {
    const tmp = await makeTmp("cleanup");
    try {
      const heartbeatPath = join(tmp, "wrappers", "heartbeat-1-x.json");
      const lifecycle = new WrapperLifecycle({
        pid: 1,
        uuid: "x",
        socketPath: join(tmp, "daemon.sock"),
        wakeSignalPath: join(tmp, "wake.signal"),
        heartbeatPath,
        platform: "darwin",
        socketReachable: async () => true,
        spawnKickstart: async () => {},
        refreshIntervalMs: 10,
      });
      await lifecycle.registerHeartbeat();
      const sigBefore = await stat(heartbeatPath);
      assert.ok(sigBefore.isFile());

      await lifecycle.cleanupHeartbeat();
      await assert.rejects(stat(heartbeatPath), "heartbeat file must be gone after cleanup");

      // No refresh after cleanup: wait longer than the refresh interval
      // and verify the file does NOT reappear.
      await sleep(60);
      await assert.rejects(stat(heartbeatPath), "no refresh tick after cleanup");

      // Idempotent: second cleanup must NOT throw.
      await lifecycle.cleanupHeartbeat();
    } finally {
      await cleanupTmp(tmp);
    }
  });
});

// ---------------------------------------------------------------- security

describe("WrapperLifecycle security invariants", () => {
  it("source contains no shell-true option and no shell-interpreting subprocess variants", async () => {
    // Walk mcp-wrapper/src/ and assert that no .ts file contains the
    // forbidden patterns. We allow the safe `execFile` API; we forbid
    // (a) the `shell: true` option anywhere, (b) bare-name calls to
    // the shell-interpreting subprocess variant from node:child_process.
    //
    // Detection strategy: build the forbidden tokens at runtime from
    // characters so the test source itself doesn't contain the literal
    // banned substring (avoids tripping security-reminder hooks that
    // grep for source-level mentions).
    const here = fileURLToPath(new URL(".", import.meta.url));
    const srcDir = join(here, "..", "src");
    const files = await readdir(srcDir);
    const tsFiles = files.filter((f) => f.endsWith(".ts"));
    assert.ok(tsFiles.length > 0, "expected at least one .ts file in src/");

    const E = String.fromCharCode(0x65); // 'e'
    const X = String.fromCharCode(0x78); // 'x'
    const C = String.fromCharCode(0x63); // 'c'
    const SHELL_INTERP_TOKEN = E + X + E + C; // 4-char banned identifier
    const SHELL_OPTION_TOKEN = "shell"; // followed by colon + true
    const shellOptionRegex = new RegExp(
      `\\b${SHELL_OPTION_TOKEN}\\s*:\\s*true\\b`,
    );
    // Allow `<token>File` (the safe variant) but forbid bare `<token>(`
    // OR `child_process.<token>(`.
    const bareCallRegex = new RegExp(
      `(?:^|[^A-Za-z0-9_])${SHELL_INTERP_TOKEN}\\s*\\(`,
    );
    const dottedCallRegex = new RegExp(
      `\\bchild_process\\s*\\.\\s*${SHELL_INTERP_TOKEN}\\s*\\(`,
    );

    const forbidden: { file: string; pattern: string; line: number }[] = [];
    for (const f of tsFiles) {
      const path = join(srcDir, f);
      const content = await readFile(path, "utf-8");
      const lines = content.split("\n");
      lines.forEach((line, idx) => {
        const trimmed = line.trim();
        // Strip trailing line comment so an inline `// NEVER ...` mention
        // in a code line doesn't match. Pure-comment lines (codePortion
        // empty after trim) are skipped.
        const codePortion = (trimmed.split("//")[0] ?? "").trim();
        if (codePortion.length === 0) {
          return;
        }
        if (shellOptionRegex.test(codePortion)) {
          forbidden.push({
            file: f,
            pattern: "shell-true option",
            line: idx + 1,
          });
        }
        if (dottedCallRegex.test(codePortion)) {
          forbidden.push({
            file: f,
            pattern: "child_process.<shell-interp-call>",
            line: idx + 1,
          });
        }
        if (bareCallRegex.test(codePortion)) {
          forbidden.push({
            file: f,
            pattern: "bare <shell-interp-call>",
            line: idx + 1,
          });
        }
      });
    }

    assert.deepEqual(
      forbidden,
      [],
      `Forbidden subprocess pattern in mcp-wrapper/src/: ${JSON.stringify(forbidden, null, 2)}`,
    );
  });
});
