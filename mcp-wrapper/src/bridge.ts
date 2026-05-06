// Phase 7.1 — pure-connector bridge. NO spawn capability.
// The daemon is launchd-managed (see scripts/install.sh).
// Wrapper connects to ~/.iai-mcp/.daemon.sock with 5s timeout.
// On connect failure, throws DaemonUnreachableError — does NOT
// attempt to spawn a daemon (eliminating Phase 7's TOCTOU race).

import * as crypto from "node:crypto";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";

// HIGH-4 LOCKED (Plan 07-04 Task 1 Step A): env override is mandatory so
// tests can isolate via tmp socket paths. The daemon-side honors the same
// env (Plan 07-02 added it to socket_server.py:serve()).
const DAEMON_SOCKET_PATH =
  process.env.IAI_DAEMON_SOCKET_PATH
  ?? path.join(os.homedir(), ".iai-mcp", ".daemon.sock");
const SOCKET_CONNECT_TIMEOUT_MS = 5000;
// 5s — covers launchd socket-activation cold-start (~3s embedder load
// + ~1s LanceDB open + buffer). launchd accepts the connection
// immediately and queues the read until the daemon is ready, so a
// single 5s timeout is sufficient even on a true cold start.
// JSON-RPC 2.0 custom server-error code (-32099..-32000 reserved by spec for
// implementation-defined server errors per jsonrpc.org/specification).
const ERR_DAEMON_UNREACHABLE = -32002;

/**
 * Phase 7.1 — clean error class thrown when the daemon socket is not
 * reachable at start(). Replaces the pre-7.1 `daemon_spawn_failed`
 * generic Error. The error message points the user at the launchd
 * recovery commands. `code` matches the existing
 * `ERR_DAEMON_UNREACHABLE` JSON-RPC server-error constant so downstream
 * consumers (handleSocketDeath in-flight rejects, `iai-mcp doctor`)
 * can pattern-match on a single numeric code.
 */
export class DaemonUnreachableError extends Error {
  public code: number;
  constructor(message: string) {
    super(message);
    this.name = "DaemonUnreachableError";
    this.code = ERR_DAEMON_UNREACHABLE;
  }
}

interface RpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params: Record<string, unknown>;
}

interface RpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
}

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

export class PythonCoreBridge {
  private sock: net.Socket | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private reconnectAttempted = false;
  // V3-05 fix: serializes the at-most-one async reconnect from
  // handleSocketDeath. Concurrent call() awaits this promise BEFORE
  // checking !this.sock so a request landing in the gap between socket
  // close and reconnect-completion does NOT reject daemon_unreachable
  // when the daemon is actually healthy.
  private reconnectPromise: Promise<void> | null = null;
  // mcp-tools-list-empty-cache fix (2026-05-02): serializes concurrent
  // start() calls. Without this, the deferred-bridge-start ordering in
  // index.ts (multiple paths can trigger start: oninitialized,
  // CallToolRequest handler, top-level fire-and-forget) would each
  // observe `this.sock === null` and race independent connectWithTimeout
  // attempts. With it, the first caller drives the connect, every other
  // caller awaits the same promise. On reject the latch clears so the
  // next start() can retry (e.g. daemon came up later).
  private startPromise: Promise<void> | null = null;
  /** V3-06: consecutive JSON.parse failures on the NDJSON stream. */
  private parseErrorStreak = 0;
  private static readonly PARSE_ERROR_REJECT_THRESHOLD = 4;

  // Allow overriding the Python interpreter via IAI_MCP_PYTHON for tests
  // that need to run the daemon against the project venv (see
  // test_mcp_tools.py).
  constructor(
    private readonly pythonCmd: string = process.env.IAI_MCP_PYTHON ?? "python3",
  ) {}

  /**
   * Phase 7.1 — pure-connector start(). Socket-only; NO spawn capability.
   * Idempotent: a second call while a socket is alive is a no-op.
   *
   * Tries to connect to ~/.iai-mcp/.daemon.sock with a 5s timeout
   * (covers launchd socket-activation cold-start). On failure, throws
   * DaemonUnreachableError pointing the user at scripts/install.sh.
   *
   * The daemon's lifecycle is owned by launchd (see
   * scripts/com.iai-mcp.daemon.plist.template); the wrapper does not
   * spawn it under any condition (eliminates Phase 7's TOCTOU race when
   * N≥3 wrappers cold-start concurrently).
   *
   * mcp-tools-list-empty-cache fix (2026-05-02): start() is now safe to
   * call concurrently from multiple async paths (top-level boot fire,
   * server.oninitialized chain, CallToolRequest lazy-await). The first
   * caller drives the actual socket connect; the rest await the shared
   * `startPromise` and observe the same outcome. On reject the latch
   * is cleared so a future call() can retry once the daemon is up.
   */
  async start(): Promise<void> {
    if (this.sock) return;  // already connected; idempotent
    if (this.startPromise) return this.startPromise;
    this.startPromise = this._doStart();
    try {
      await this.startPromise;
    } catch (err) {
      // Allow a future caller to retry — the daemon may simply have been
      // slow to come up. Without clearing the latch, every subsequent
      // start() would short-circuit on the rejected memoised promise.
      this.startPromise = null;
      throw err;
    }
    // On success, leave startPromise set; further calls short-circuit on
    // `this.sock` truthiness (set inside _doStart before resolution).
  }

  private async _doStart(): Promise<void> {
    // Reset reconnect-once latch so a fresh start() (e.g. after explicit
    // disconnect) is treated as a new session by handleSocketDeath.
    this.reconnectAttempted = false;

    let sock: net.Socket;
    try {
      sock = await this.connectWithTimeout(
        DAEMON_SOCKET_PATH,
        SOCKET_CONNECT_TIMEOUT_MS,
      );
    } catch (e) {
      throw new DaemonUnreachableError(
        "iai-mcp daemon not running. "
        + "Run: launchctl load -w ~/Library/LaunchAgents/com.iai-mcp.daemon.plist "
        + "or run scripts/install.sh"
      );
    }
    this.sock = sock;
    this.attachSocketHandlers();
  }

  /**
   * Promise wrapper around net.createConnection with a hard timeout.
   * Adapted from emitSessionOpen (lines below) — same silent-fail safety
   * pattern, but resolves with the live socket on success so the caller
   * can retain it for long-lived JSON-RPC traffic.
   */
  private connectWithTimeout(
    socketPath: string,
    timeoutMs: number,
  ): Promise<net.Socket> {
    return new Promise((resolve, reject) => {
      const sock = net.createConnection(socketPath);
      const t = setTimeout(() => {
        try { sock.destroy(); } catch { /* ignore */ }
        reject(new Error("connect_timeout"));
      }, timeoutMs);
      sock.once("connect", () => {
        clearTimeout(t);
        resolve(sock);
      });
      sock.once("error", (e) => {
        clearTimeout(t);
        reject(e);
      });
    });
  }

  private attachSocketHandlers(): void {
    if (!this.sock) return;
    this.sock.on("data", (chunk: Buffer) => this.handleData(chunk));
    this.sock.on("close", () => this.handleSocketDeath("closed"));
    this.sock.on("error", (e: Error) => this.handleSocketDeath(`error: ${e.message}`));
  }

  /**
   * NDJSON read buffer: socket data arrives in arbitrary chunks; we buffer
   * + split on `\n` manually. Each complete line is one JSON-RPC response
   * envelope.
   */
  private handleData(chunk: Buffer): void {
    this.buffer += chunk.toString("utf-8");
    let nl: number;
    while ((nl = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      if (!line) continue;
      this.handleLine(line);
    }
  }

  private handleLine(line: string): void {
    let msg: RpcResponse;
    try {
      msg = JSON.parse(line) as RpcResponse;
    } catch {
      this.parseErrorStreak += 1;
      if (
        this.parseErrorStreak >= PythonCoreBridge.PARSE_ERROR_REJECT_THRESHOLD
        && this.pending.size > 0
      ) {
        const oldestId = Math.min(...this.pending.keys());
        const handler = this.pending.get(oldestId);
        if (handler) {
          this.pending.delete(oldestId);
          handler.reject(
            new Error(
              `parse_error: ${PythonCoreBridge.PARSE_ERROR_REJECT_THRESHOLD} consecutive non-JSON lines on daemon socket; rejecting stale RPC id=${oldestId}`,
            ),
          );
        }
        try {
          process.stderr.write(
            `${JSON.stringify({
              event: "bridge_ndjson_parse_error_streak",
              threshold: PythonCoreBridge.PARSE_ERROR_REJECT_THRESHOLD,
              rejected_rpc_id: oldestId,
            })}\n`,
          );
        } catch { /* ignore */ }
        this.parseErrorStreak = 0;
      }
      return; // non-JSON line -- ignore (e.g., stray prints from daemon libs)
    }
    this.parseErrorStreak = 0;
    const handler = this.pending.get(msg.id);
    if (!handler) return;
    this.pending.delete(msg.id);
    if (msg.error) {
      handler.reject(new Error(msg.error.message));
    } else {
      handler.resolve(msg.result);
    }
  }

  /**
   * R5 fail-loud: socket close/error rejects ALL pending Promises with
   * `daemon_unreachable` (-32002). D7-04 / SPEC R5: ONE reconnect attempt
   * (catches launchd KeepAlive respawn windows). After that attempt the
   * bridge stays degraded — every subsequent call returns
   * `daemon_unreachable` until the wrapper itself restarts.
   */
  private handleSocketDeath(why: string): void {
    // Synchronous: every pending request fails LOUD immediately so callers
    // see daemon_unreachable instead of hanging forever (D7-04 / SPEC R5).
    const err = new Error(`daemon_unreachable: socket ${why} (code ${ERR_DAEMON_UNREACHABLE})`);
    for (const [, p] of this.pending) p.reject(err);
    this.pending.clear();
    this.sock = null;
    // Clear the start-latch so a future call() can retry start() (e.g.
    // after launchd respawn). reconnectPromise (below) handles the
    // immediate one-shot reconnect; startPromise reset enables
    // long-tail retry from any new caller after that.
    this.startPromise = null;

    if (this.reconnectAttempted) return;
    this.reconnectAttempted = true;

    // Async reconnect-once. Concurrent call() awaits this promise BEFORE
    // checking !this.sock, eliminating the V3-05 race.
    this.reconnectPromise = (async () => {
      try {
        // Test-only deterministic widener for the V3-05 race window.
        // In production this env var is unset → 0 ms → no-op. The
        // V3-05 regression test (tests/test_socket_disconnect_reconnect.py)
        // sets IAI_MCP_RECONNECT_TEST_DELAY_MS=1000 so the racing
        // call() can land deterministically inside the gap between
        // socket close and reconnect-completion. Without this delay the
        // race window is sub-millisecond and the regression test cannot
        // distinguish pre-fix (rejects daemon_unreachable) from post-fix
        // (awaits reconnectPromise, succeeds).
        const testDelayMs = Number(
          process.env.IAI_MCP_RECONNECT_TEST_DELAY_MS ?? "0",
        );
        if (testDelayMs > 0) {
          await new Promise<void>((r) => setTimeout(r, testDelayMs));
        }
        // Manually do socket-first connect (without resetting the latch
        // that start() does) so a SECOND mid-call death stays degraded.
        this.sock = await this.connectWithTimeout(
          DAEMON_SOCKET_PATH,
          SOCKET_CONNECT_TIMEOUT_MS,
        );
        this.attachSocketHandlers();
      } catch {
        // stay degraded — every subsequent call sees this.sock === null
        // and rejects with daemon_unreachable.
      } finally {
        this.reconnectPromise = null;
      }
    })();
  }

  /**
   * Send a JSON-RPC 2.0 request over the socket; resolves with `result`
   * or rejects with the daemon-side `error.message`.
   *
   * R5 fail-loud: when this.sock is null (post-death, post-disconnect,
   * pre-start) the call rejects synchronously with `daemon_unreachable`.
   * NO silent fallback to a local Python core spawn.
   */
  async call<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T> {
    // V3-05 fix: if a reconnect is in flight, wait for it before deciding
    // whether the socket is alive. Without this await, a call() landing in
    // the gap between socket close and reconnect-completion would reject
    // with daemon_unreachable even though the daemon is healthy.
    if (this.reconnectPromise) {
      await this.reconnectPromise;
    }
    if (!this.sock) {
      throw new Error(`daemon_unreachable: bridge not connected (code ${ERR_DAEMON_UNREACHABLE})`);
    }
    const id = this.nextId++;
    const req: RpcRequest = { jsonrpc: "2.0", id, method, params };
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      try {
        this.sock!.write(JSON.stringify(req) + "\n");
      } catch (e) {
        this.pending.delete(id);
        reject(e as Error);
      }
    });
  }

  /**
   * Public API: close the socket but leave the daemon running.
   * Used by index.ts SIGTERM/SIGINT handlers.
   *
   * After Phase 7 the wrapper does NOT own the daemon's lifecycle —
   * disconnecting a wrapper must NOT kill the singleton, otherwise other
   * wrappers (other MCP hosts, sub-agents) would lose their
   * shared brain.
   */
  disconnect(): void {
    if (this.sock) {
      try { this.sock.end(); } catch { /* ignore */ }
      try { this.sock.destroy(); } catch { /* ignore */ }
      this.sock = null;
    }
    // Clear the start-latch so a fresh start() (e.g. test re-use of the
    // bridge instance) is treated as a brand new connection.
    this.startPromise = null;
    // Reject any in-flight calls with a clean message (NOT
    // daemon_unreachable — the daemon is fine; we just chose to close).
    for (const [, p] of this.pending) {
      p.reject(new Error("bridge_disconnected"));
    }
    this.pending.clear();
  }

  // Visible for tests: smoke endpoint replacing the pre-Phase-7
  // isRunning() that checked for a child process.
  isConnected(): boolean {
    return this.sock !== null;
  }
}


// ---------------------------------------------------------------------------
// Plan 05-04 TOK-14 / D5-05 — session_open emit over the daemon unix socket.
// UNCHANGED by Phase 7 (Plan 07-04). Same socket path; brief separate
// connection that fires a one-shot HIPPEA pre-warm hint then closes.
// ---------------------------------------------------------------------------


/**
 * Path to the Python daemon's unix control socket.
 * Mirror of `concurrency.SOCKET_PATH` in the Python core (`~/.iai-mcp/.daemon.sock`).
 *
 * Honors `IAI_DAEMON_SOCKET_PATH` so tests can isolate via tmp socket paths
 * (matches the same env override the main bridge socket connect uses).
 */
export function sessionOpenSocketPath(): string {
  const env = process.env.IAI_DAEMON_SOCKET_PATH;
  if (env) return env;
  return path.join(os.homedir(), ".iai-mcp", ".daemon.sock");
}


/**
 * Generate a fresh session identifier for the boot event.
 * Node stdlib since 14.17 — no dependency added.
 */
export function newSessionId(): string {
  return crypto.randomUUID();
}


/**
 * Fire-and-forget NDJSON `session_open` message to the daemon socket.
 *
 * Contract:
 *  - Writes one line: `{"type":"session_open","session_id":"...","ts":"..."}\n`
 *  - One-shot semantics: does **not** read the daemon's response bytes before
 *    `end()` — intentional (HIPPEA hint only). If the daemon wrote backpressure
 *    or error bytes, they are left unread; the separate long-lived `PythonCoreBridge`
 *    connection owns JSON-RPC traffic.
 *  - Silent-fail on any network, socket-not-found, or timeout error. The
 *    Python core's `_first_turn_recall_hook` falls back to the cold recall
 *    path when the cascade LRU is empty (expected when daemon is down).
 *  - Hard timeout at 2s so a hung socket cannot delay wrapper boot.
 *
 * Returns a Promise<void> that ALWAYS resolves (never rejects) so callers
 * can use `void emitSessionOpen(...)` in a sync bootstrap block without
 * an explicit `.catch`.
 */
export function emitSessionOpen(sessionId: string): Promise<void> {
  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    try {
      const socketPath = sessionOpenSocketPath();
      const sock = net.createConnection(socketPath, () => {
        const msg =
          JSON.stringify({
            type: "session_open",
            session_id: sessionId,
            ts: new Date().toISOString(),
          }) + "\n";
        sock.write(msg, () => {
          sock.end();
        });
      });
      sock.on("error", () => finish());
      sock.on("close", () => finish());
      sock.setTimeout(2000, () => {
        try {
          sock.destroy();
        } catch {
          // ignore
        }
        finish();
      });
    } catch {
      // Any sync setup failure -> silent fallback.
      finish();
    }
  });
}
