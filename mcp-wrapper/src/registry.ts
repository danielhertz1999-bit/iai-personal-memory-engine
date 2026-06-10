
import { TOOL_NAMES, toolSchemas, type ToolName } from "./tools.js";

export const HOT_TOOLS: readonly ToolName[] = [...TOOL_NAMES] as const;

export const CONTEXT_EDITING_CONFIG = {
  type: "clear_tool_uses_20250919" as const,
  trigger: {
    type: "input_tokens" as const,
    value: 30_000,
  },
} as const;

export function listHotTools() {
  return HOT_TOOLS.map((n) => toolSchemas[n]);
}

export async function loadColdTool(_name: string): Promise<unknown | null> {
  return null;
}
