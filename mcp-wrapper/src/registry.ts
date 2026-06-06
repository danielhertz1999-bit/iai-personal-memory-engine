// Lazy tool registry + context-editing config.
//
// All tools are hot (small enough to always keep resident). The `loadColdTool`
// hook exists as an extension point for future cold tools registered here and
// looked up lazily by the MCP host's ToolSearch extension.
//
// Context editing: we advertise `clear_tool_uses_20250919` with a 30k-token
// trigger. When the context crosses 30k tokens the Anthropic API will drop
// earlier tool_use / tool_result messages, freeing headroom for continued
// reasoning without reloading the full session prefix.
//
// Exact shape per Anthropic's context-management docs -- these strings are
// consumed verbatim by the API.

import { TOOL_NAMES, toolSchemas, type ToolName } from "./tools.js";

// hot tools: all 5 always-resident (fixed surface).
// Iteration order matches TOOL_NAMES so tools/list is deterministic.
export const HOT_TOOLS: readonly ToolName[] = [...TOOL_NAMES] as const;

/** Anthropic context-editing config -- exact shape consumed by the API.
 *
 * `clear_tool_uses_20250919` is the context-edit strategy Anthropic released
 * on 2025-09-19; the trigger pairs `type: "input_tokens"` with a numeric
 * threshold. The 30k-token threshold provides empirically enough headroom to
 * preserve ~8-10 turns of tool exchange before trimming. */
export const CONTEXT_EDITING_CONFIG = {
  type: "clear_tool_uses_20250919" as const,
  trigger: {
    type: "input_tokens" as const,
    value: 30_000,
  },
} as const;

/** Return the full tool-schema objects for the hot tools.
 *
 * MCP `tools/list` handler calls this directly. Kept as a function rather
 * than a const array so future versions can mutate the returned shape
 * (e.g., swap in per-user personalised descriptions) without changing the
 * call site. */
export function listHotTools() {
  return HOT_TOOLS.map((n) => toolSchemas[n]);
}

/** Hook: lazy-load a tool that isn't in HOT_TOOLS.
 *
 * Always returns null -- the MCP host's ToolSearch extension will fall back
 * to HOT_TOOLS when this returns null. */
export async function loadColdTool(_name: string): Promise<unknown | null> {
  return null;
}
