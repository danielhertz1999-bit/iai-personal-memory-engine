// Wrapper-side sick-notification: on session-start, spawn `iai-mcp doctor`
// and write one line to stderr if it reports FAIL. Injectable spawnFn,
// short timeout, silent on spawn failure.
//
// Exit-code semantics from `iai-mcp doctor`:
// 0 = all PASS
// 1 = >=1 FAIL without --apply
// 2 = --apply ran but final re-check still has FAIL
// We treat any non-zero exit as "sick"; null = probe itself failed
// (spawn error / timeout) and is silent so we never double-warn on top
// of the daemon_unreachable surface that index.ts already produces.

import { spawn, type SpawnOptions } from "node:child_process";

const DOCTOR_PROBE_TIMEOUT_MS = 5_000;

/**
 * Spawn `iai-mcp doctor` and resolve to its exit code (or null on spawn
 * failure / timeout). Does not parse stdout — the exit code is the entire
 * trust input.
 *
 * @param spawnFn injectable child_process.spawn (test seam)
 * @param timeoutMs probe kill-deadline in ms
 */
export async function probeDaemonDoctor(
  spawnFn: typeof spawn = spawn,
  timeoutMs: number = DOCTOR_PROBE_TIMEOUT_MS,
): Promise<number | null> {
  const cli = process.env.IAI_MCP_CLI ?? "iai-mcp";
  const args = ["doctor"];
  return new Promise<number | null>((resolve) => {
    // stdio "ignore" everywhere: we only key off exit code. Doctor's own
    // stdout/stderr is for direct CLI invocation; the wrapper warning is
    // intentionally a separate, single-line summary.
    const opts: SpawnOptions = { stdio: ["ignore", "ignore", "ignore"] };
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawnFn(cli, args, opts);
    } catch {
      resolve(null);
      return;
    }
    const t = setTimeout(() => {
      try {
        proc.kill();
      } catch {
        // best-effort kill; the child may already be gone
      }
      resolve(null);
    }, timeoutMs);
    proc.on("error", () => {
      clearTimeout(t);
      resolve(null);
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(t);
      resolve(code);
    });
  });
}

/**
 * If `exitCode` indicates a sick daemon (1 or 2), write exactly one line
 * to stderr describing the state. Silent on 0 / null / negative codes.
 *
 * Message is a fixed template with the exit code interpolated — no
 * environment-variable or stdout text is interpolated, so log-injection
 * surface is bounded to a single integer.
 */
export function emitSickWarningIfNeeded(exitCode: number | null): void {
  if (exitCode === null || exitCode === 0) {
    return;
  }
  const msg =
    `iai-mcp warning: daemon doctor reports FAIL (exit=${exitCode}). ` +
    "Run `iai-mcp doctor` for details. Memory tools may fall back to bank-recall.\n";
  try {
    process.stderr.write(msg);
  } catch {
    // stderr unavailable (extremely unlikely in MCP context); silent fail.
  }
}
