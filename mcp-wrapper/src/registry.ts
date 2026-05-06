// Lazy tool registry + context-editing config (TOK-02, TOK-05).
//
// TOK-02 ToolSearch lazy-load: in Phase 1 all 5 Phase-1 tools are hot (small
// enough to always keep resident). The `loadColdTool` hook exists as a Phase-2
// extension point -- when Mem-08 / schema_list / curiosity_pending ship (Phase
// 2), they'll register here and be looked up
// lazily by the MCP host's ToolSearch extension.
//
// TOK-05 context editing: we advertise `clear_tool_uses_20250919` with a
// 30k-token trigger. When Claude's context crosses 30k tokens the Anthropic
// API will drop earlier tool_use / tool_result messages, freeing headroom
// for continued reasoning without reloading the full session prefix.
//
// Exact shape per Anthropic's context-management docs -- these strings are
// consumed verbatim by the API.

import { TOOL_NAMES, toolSchemas, type ToolName } from "./tools.js";

// Phase-1 hot tools: all 5 always-resident (D-12 fixed surface).
// Iteration order matches TOOL_NAMES so tools/list is deterministic.
export const HOT_TOOLS: readonly ToolName[] = [...TOOL_NAMES] as const;

/** TOK-05 Anthropic context-editing config -- exact shape consumed by the API.
 *
 *  `clear_tool_uses_20250919` is the dated context-edit strategy Anthropic
 *  released on 2025-09-19; the trigger pairs `type: "input_tokens"` with a
 *  numeric threshold that fires the edit. D-10 puts the threshold at 30k
 *  tokens -- empirically enough headroom to preserve ~8-10 turns of tool
 *  exchange before trimming. */
export const CONTEXT_EDITING_CONFIG = {
  type: "clear_tool_uses_20250919" as const,
  trigger: {
    type: "input_tokens" as const,
    value: 30_000,
  },
} as const;

/** Return the full tool-schema objects for the hot tools.
 *
 *  MCP `tools/list` handler calls this directly. Kept as a function rather
 *  than a const array so future versions can mutate the returned shape
 *  (e.g., swap in per-user personalised descriptions) without changing the
 *  call site. */
export function listHotTools() {
  return HOT_TOOLS.map((n) => toolSchemas[n]);
}

/** Phase-2 hook: lazy-load a tool that isn't in HOT_TOOLS.
 *
 *  Phase 1 always returns null -- the MCP host's ToolSearch extension will
 *  fall back to HOT_TOOLS when this returns null, which is exactly what we
 *  want. Phase 2 populates this with a dynamic import of the new tool's
 *  schema module. */
export async function loadColdTool(_name: string): Promise<unknown | null> {
  return null;
}
