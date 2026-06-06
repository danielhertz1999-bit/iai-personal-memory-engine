// Anthropic 1h-TTL prompt caching.
//
// Single breakpoint at the stable/volatile boundary. The Python core's
// `session_start_payload` returns the 4-segment cached prefix; this module
// wraps it in Anthropic `content` blocks and stamps `cache_control` on the
// last stable block so Anthropic's cache sees one hashable suffix.
//
// cache_control TTL="1h" is the Anthropic prompt-caching extended-TTL option
// released in Oct 2024 (enabled per-org; falls back to "5m" default when
// unsupported). Rationale per: session-start prefix rarely changes
// within an hour, so 1h TTL hits Anthropic's cache on every turn after the
// first fresh-session write (8000-token premium absorbed once).

export interface CacheControl {
  readonly type: "ephemeral";
  readonly ttl: "1h" | "5m";
}

export interface ContentBlock {
  type: string;
  text?: string;
  cache_control?: CacheControl;
}

export interface SessionPayloadRaw {
  l0: string;
  l1: string;
  l2: string[];
  rich_club: string;
  total_cached_tokens: number;
  total_dynamic_tokens: number;
  breakpoint_marker?: string;
}

/** Attach a single `cache_control` breakpoint at the stable/volatile boundary.
 *
 * Emits exactly one breakpoint: on the LAST block of `stable`.
 * If `stable` is empty the function returns the volatile blocks unchanged --
 * there is no sensible place to hang a breakpoint on an empty prefix and
 * Anthropic's API would reject the request.
 *
 * Returns a new array; inputs are not mutated. */
export function applyCacheBreakpoint(
  stable: ContentBlock[],
  volatile: ContentBlock[],
): ContentBlock[] {
  if (stable.length === 0) {
    return [...volatile];
  }
  const cloned = stable.map((b) => ({ ...b }));
  cloned[cloned.length - 1] = {
    ...cloned[cloned.length - 1],
    cache_control: { type: "ephemeral", ttl: "1h" },
  };
  return [...cloned, ...volatile];
}

/** Build the cached system prompt from the Python session_start_payload.
 *
 * Segments in order: L0 identity, L1 critical facts, L2 community summaries
 * (one block per community), rich-club prefetch. Empty segments are skipped
 * so the cache-key is stable across sessions where, say, L1 is empty.
 *
 * Returned blocks already have the cache_control breakpoint applied. */
export function buildCachedSystemPrompt(
  payload: SessionPayloadRaw,
): ContentBlock[] {
  const stable: ContentBlock[] = [];
  if (payload.l0) {
    stable.push({ type: "text", text: `# L0 identity\n${payload.l0}` });
  }
  if (payload.l1) {
    stable.push({ type: "text", text: `# L1 critical facts\n${payload.l1}` });
  }
  for (const segment of payload.l2) {
    stable.push({ type: "text", text: `# L2 community\n${segment}` });
  }
  if (payload.rich_club) {
    stable.push({
      type: "text",
      text: `# Global rich-club\n${payload.rich_club}`,
    });
  }
  return applyCacheBreakpoint(stable, []);
}
