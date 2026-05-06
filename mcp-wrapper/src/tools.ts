// Phase-1 (D-12) + Plan 02-04 (MCP-05/07/08) + Plan 03 (CONN-05/07 + AUTIST-13) tools.
//
// Tool shapes are JSON-schema dicts consumable by the MCP SDK's ListTools
// handler. Descriptions are written for Claude's tool-discovery heuristics
// (concise, task-oriented, reference the autistic-kernel defaults where they
// affect behaviour).
//
// Plan 02-04 adds 3 user-introspection tools:
// - curiosity_pending  (MCP-07): list pending curiosity questions
// - schema_list        (MCP-08): list induced schemas
// - events_query       (MCP-05): user-visible events audit
//
// Plan 03 adds 3 scientific-depth tools:
// - memory_recall_structural (CONN-05): TEM role->filler structural recall
// - topology                 (CONN-07): Ashby sigma diagnostic snapshot
// - camouflaging_status      (AUTIST-13): ecological self-regulation status

import type { PythonCoreBridge } from "./bridge.js";

export const TOOL_NAMES = [
  "memory_recall",
  "memory_recall_structural",
  "memory_reinforce",
  "memory_contradict",
  "memory_capture",
  "memory_consolidate",
  "profile_get_set",
  "curiosity_pending",
  "schema_list",
  "events_query",
  "topology",
  "camouflaging_status",
] as const;

export type ToolName = (typeof TOOL_NAMES)[number];

interface ToolSchema {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

export const toolSchemas: Record<ToolName, ToolSchema> = {
  memory_recall: {
    name: "memory_recall",
    description:
      "Recall verbatim memories matching cue. Returns hits + anti_hits.",
    inputSchema: {
      type: "object",
      properties: {
        cue: {
          type: "string",
          description: "Natural-language query to match against stored memories.",
        },
        budget_tokens: {
          type: "integer",
          description: "Soft token budget for response (default 1500).",
          default: 1500,
        },
        session_id: {
          type: "string",
          description:
            "Current session id; gets written into every recalled record's provenance (MEM-05).",
        },
        cue_embedding: {
          type: "array",
          items: { type: "number" },
          description:
            "Optional pre-computed embedding vector for the cue " +
            "(EMBED_DIM=384 floats; bge-small-en-v1.5). " +
            "When omitted, the daemon embeds the cue server-side. " +
            "Used by memory_contradict and tests that need byte-stable embeddings.",
        },
        language: {
          type: "string",
          description:
            "Optional ISO-639-1 language hint for the sleep-suggestion path " +
            "(8 supported: en/ru/ja/ar/de/fr/es/zh). Defaults to 'en' " +
            "when omitted. Hot-path retrieval is language-agnostic; this " +
            "key only affects the sleep-suggestion regex pre-screen.",
        },
      },
      required: ["cue"],
    },
  },
  memory_reinforce: {
    name: "memory_reinforce",
    description:
      "Boost Hebbian edges among co-retrieved record ids.",
    inputSchema: {
      type: "object",
      properties: {
        ids: {
          type: "array",
          items: { type: "string", format: "uuid" },
          description: "Record UUIDs that were co-retrieved in the current context.",
        },
      },
      required: ["ids"],
    },
  },
  memory_contradict: {
    name: "memory_contradict",
    description:
      "Mark a record contradicted; new fact stored as new record.",
    inputSchema: {
      type: "object",
      properties: {
        id: {
          type: "string",
          format: "uuid",
          description: "UUID of the record being contradicted.",
        },
        new_fact: {
          type: "string",
          description: "The updated verbatim fact. Stored as a new record.",
        },
        cue_embedding: {
          type: "array",
          items: { type: "number" },
          description:
            "Optional pre-computed embedding vector for the contradicting " +
            "fact (EMBED_DIM=384 floats; bge-small-en-v1.5). When omitted, " +
            "the daemon embeds new_fact server-side.",
        },
      },
      required: ["id", "new_fact"],
    },
  },
  memory_capture: {
    name: "memory_capture",
    description:
      "Capture a verbatim turn. Auto-dedups at cos>=0.95 (reinforces). " +
      "Use for corrections + load-bearing decisions.",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description:
            "Verbatim text to capture (user utterance, Claude decision, or observation). " +
            "Min 12 chars, max 8000 (longer is truncated).",
        },
        cue: {
          type: "string",
          description:
            "Short natural-language cue used for embedding + dedup lookup. " +
            "If empty, `text` itself is embedded.",
        },
        tier: {
          type: "string",
          enum: ["working", "episodic", "semantic", "procedural", "parametric"],
          default: "episodic",
          description:
            "Memory tier. Default 'episodic' (verbatim user utterances). " +
            "Use 'semantic' for induced summaries, 'procedural' for learned behaviour notes.",
        },
        session_id: {
          type: "string",
          description: "Current session id for provenance (MEM-05).",
        },
        role: {
          type: "string",
          enum: ["user", "assistant", "system"],
          default: "user",
          description: "Who produced this turn — tags the record for filtering.",
        },
      },
      required: ["text"],
    },
  },
  memory_consolidate: {
    name: "memory_consolidate",
    description:
      "Trigger memory consolidation.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description:
            "Optional session id used for provenance tagging on the " +
            "consolidate event. Defaults to '-' when omitted.",
        },
      },
    },
  },
  profile_get_set: {
    name: "profile_get_set",
    description:
      "Read or write a profile knob (11 sealed: 10 AUTIST + wake_depth). operation: get|set.",
    inputSchema: {
      type: "object",
      properties: {
        operation: {
          type: "string",
          enum: ["get", "set"],
          description: "Whether to read or write a knob.",
        },
        knob: {
          type: "string",
          description: "Knob name. Omit on 'get' to retrieve all live + deferred knobs.",
        },
        value: {
          description: "New value when operation='set'. Any JSON-serialisable type.",
        },
      },
      required: ["operation"],
    },
  },
  curiosity_pending: {
    name: "curiosity_pending",
    description:
      "List pending curiosity questions. Optional session_id filter.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "Only return questions from this session.",
        },
      },
    },
  },
  schema_list: {
    name: "schema_list",
    description:
      "List induced schemas. Optional domain + confidence_min filters.",
    inputSchema: {
      type: "object",
      properties: {
        domain: {
          type: "string",
          description: "Only return schemas tagged with this domain (e.g. 'coding').",
        },
        confidence_min: {
          type: "number",
          description: "Minimum parsed confidence (0.0-1.0). Default 0.0.",
          default: 0.0,
        },
      },
    },
  },
  events_query: {
    name: "events_query",
    description:
      "Query user-visible events by kind, since, severity, limit.",
    inputSchema: {
      type: "object",
      properties: {
        kind: {
          type: "string",
          description:
            "Event kind. Must be in the whitelist (see tool description).",
        },
        since: {
          type: "string",
          description: "ISO-8601 timestamp; only events at or after this are returned.",
        },
        severity: {
          type: "string",
          enum: ["info", "warning", "critical"],
          description: "Optional severity filter.",
        },
        limit: {
          type: "integer",
          description: "Maximum events returned (default 100, capped at 1000).",
          default: 100,
        },
      },
      required: ["kind"],
    },
  },
  memory_recall_structural: {
    name: "memory_recall_structural",
    description:
      "Structural recall via role-filler bindings (TEM). O(N) scan; max_records caps.",
    inputSchema: {
      type: "object",
      properties: {
        structure_query: {
          type: "object",
          description:
            "Optional role->filler map, e.g. {\"agent\": \"Alice\"}. Each value is hashed to a filler hypervector. When omitted or empty, query HV is zero-filled and every row with structure_hv is scored (expensive at large N).",
          additionalProperties: { type: "string" },
        },
        budget_tokens: {
          type: "integer",
          description: "Soft token budget for response (default 2000).",
          default: 2000,
        },
        max_records: {
          type: "integer",
          description:
            "Hard cap on records scanned after fetch (default 5000, max 50000). Prevents accidental full-corpus scans from `{}`.",
          default: 5000,
        },
      },
      required: [],
    },
  },
  topology: {
    name: "topology",
    description:
      "Topology snapshot: N, C, L, sigma, community_count, regime.",
    inputSchema: { type: "object", properties: {} },
  },
  camouflaging_status: {
    name: "camouflaging_status",
    description:
      "Camouflaging detection status; window_size weekly points.",
    inputSchema: {
      type: "object",
      properties: {
        window_size: {
          type: "integer",
          description: "Weekly points in the sliding window (default 5).",
          default: 5,
        },
      },
    },
  },
};

export async function invokeTool(
  bridge: PythonCoreBridge,
  name: ToolName,
  args: Record<string, unknown>,
): Promise<unknown> {
  switch (name) {
    case "memory_recall":
      return bridge.call("memory_recall", args);
    case "memory_reinforce":
      return bridge.call("memory_reinforce", args);
    case "memory_contradict":
      return bridge.call("memory_contradict", args);
    case "memory_capture":
      return bridge.call("memory_capture", args);
    case "memory_consolidate":
      return bridge.call("memory_consolidate", args);
    case "profile_get_set": {
      const op = args.operation as string;
      if (op === "get") {
        return bridge.call("profile_get", { knob: args.knob ?? null });
      }
      if (op === "set") {
        return bridge.call("profile_set", {
          knob: args.knob,
          value: args.value,
        });
      }
      throw new Error(`unknown operation ${op}`);
    }
    case "curiosity_pending":
      return bridge.call("curiosity_pending", args);
    case "schema_list":
      return bridge.call("schema_list", args);
    case "events_query":
      return bridge.call("events_query", args);
    case "memory_recall_structural":
      return bridge.call("memory_recall_structural", args);
    case "topology":
      return bridge.call("topology", args);
    case "camouflaging_status":
      return bridge.call("camouflaging_status", args);
  }
}
