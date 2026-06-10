
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
