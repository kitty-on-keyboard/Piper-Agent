import {
  ApprovalRequestNotification,
  ChecklistNotification,
  ModelStatusNotification,
  NoticeNotification,
  PerfNotification,
  PerfSample,
  PlanReadyNotification,
  ResumeResult,
  RunEndNotification,
  TokenNotification,
  TurnNotification,
  VerificationNotification,
} from "../protocol.generated";

export interface ToolCallItem {
  id: string;
  tool_name: string;
  tool_args: string;
  tool_status: string;
  summary: string;
  duration_ms: number;
  read_bytes: number;
  edit_bytes: number;
  truncated?: boolean;
}

export interface ChecklistRow {
  text: string;
  done: boolean;
}

export type ApprovalResolution = "approved" | "denied";

export type FeedEntry =
  | { id: string; kind: "user"; text: string; images?: string[] }
  | { id: string; kind: "think"; text: string; startedAtMs: number; endedAtMs?: number }
  | { id: string; kind: "assistant"; text: string; streaming: boolean }
  | { id: string; kind: "tools"; calls: ToolCallItem[] }
  | {
      id: string;
      kind: "approval";
      request: ApprovalRequestNotification;
      resolution?: ApprovalResolution;
    }
  | { id: string; kind: "plan"; plan: string }
  | { id: string; kind: "checklist"; items: ChecklistRow[] }
  | { id: string; kind: "verification"; data: VerificationNotification }
  | { id: string; kind: "notice"; data: NoticeNotification }
  | { id: string; kind: "omitted"; count: number };

export interface SessionState {
  activeRunId: string | null;
  running: boolean;
  entries: FeedEntry[];
  perf: PerfSample | null;
  lastRunEnd: RunEndNotification | null;
  modelStatus: ModelStatusNotification;
}

export type SessionAction =
  | { type: "reset" }
  | { type: "user_start"; runId: string; text: string; images?: string[] }
  | { type: "user_followup"; runId: string; text: string; images?: string[]; steering: boolean }
  | { type: "token"; note: TokenNotification; nowMs: number }
  | { type: "turn"; note: TurnNotification; nowMs: number }
  | { type: "checklist"; note: ChecklistNotification }
  | { type: "plan_ready"; note: PlanReadyNotification }
  | { type: "verification"; note: VerificationNotification }
  | { type: "approval_request"; note: ApprovalRequestNotification }
  | { type: "approval_resolved"; requestId: string; resolution: ApprovalResolution }
  | { type: "perf"; note: PerfNotification }
  | { type: "notice"; note: NoticeNotification }
  | { type: "run_end"; note: RunEndNotification; nowMs: number }
  | { type: "model_status"; note: ModelStatusNotification }
  | { type: "resumed"; res: ResumeResult }
  | { type: "local_error"; text: string };

let nextItemSeq = 1;
function uid(prefix: string): string {
  return `${prefix}_${nextItemSeq++}`;
}

export function createInitialSessionState(): SessionState {
  return {
    activeRunId: null,
    running: false,
    entries: [],
    perf: null,
    lastRunEnd: null,
    modelStatus: {
      state: "unloaded",
      model_dir: "",
      detail: "",
      elapsed_ms: 0,
      supports_reasoning_effort: false,
    },
  };
}

function finalizeStreamingEntries(entries: FeedEntry[], nowMs: number): FeedEntry[] {
  return entries.map((entry) => {
    if (entry.kind === "think" && entry.endedAtMs === undefined) {
      return { ...entry, endedAtMs: nowMs };
    }
    if (entry.kind === "assistant" && entry.streaming) {
      return { ...entry, streaming: false };
    }
    return entry;
  });
}

function parseChecklistJson(raw: string): ChecklistRow[] {
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((row): ChecklistRow | null => {
        if (typeof row === "string") return { text: row, done: false };
        if (!row || typeof row !== "object") return null;
        const r = row as Record<string, unknown>;
        const text =
          typeof r.text === "string"
            ? r.text
            : typeof r.title === "string"
              ? r.title
              : typeof r.label === "string"
                ? r.label
                : "";
        if (!text) return null;
        const done = r.done === true || r.completed === true || r.status === "done";
        return { text, done };
      })
      .filter((x): x is ChecklistRow => x !== null);
  } catch {
    return [];
  }
}

export function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case "reset":
      return {
        ...state,
        activeRunId: null,
        running: false,
        entries: [],
        perf: null,
        lastRunEnd: null,
      };

    case "model_status":
      return { ...state, modelStatus: action.note };

    case "user_start":
      return {
        ...state,
        activeRunId: action.runId,
        running: true,
        lastRunEnd: null,
        entries: [{ id: uid("u"), kind: "user", text: action.text, images: action.images }],
      };

    case "user_followup": {
      const entries = action.steering
        ? state.entries
        : finalizeStreamingEntries(state.entries, Date.now());
      return {
        ...state,
        activeRunId: action.runId || state.activeRunId,
        running: true,
        lastRunEnd: null,
        entries: [
          ...entries,
          { id: uid("u"), kind: "user", text: action.text, images: action.images },
        ],
      };
    }

    case "token": {
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      const chunk = action.note.text;
      if (!chunk) return state;

      const entries = [...state.entries];
      const last = entries[entries.length - 1];
      const isThink = action.note.channel === "think";

      if (isThink) {
        if (last && last.kind === "think" && last.endedAtMs === undefined) {
          entries[entries.length - 1] = { ...last, text: last.text + chunk };
        } else {
          entries.push({
            id: uid("th"),
            kind: "think",
            text: chunk,
            startedAtMs: action.nowMs,
          });
        }
        return { ...state, entries };
      }

      // Leaving think -> close open think block cleanly so its elapsed time freezes.
      if (last && last.kind === "think" && last.endedAtMs === undefined) {
        entries[entries.length - 1] = { ...last, endedAtMs: action.nowMs };
      }
      const currentLast = entries[entries.length - 1];
      if (currentLast && currentLast.kind === "assistant" && currentLast.streaming) {
        entries[entries.length - 1] = { ...currentLast, text: currentLast.text + chunk };
      } else {
        entries.push({ id: uid("a"), kind: "assistant", text: chunk, streaming: true });
      }
      return { ...state, entries };
    }

    case "turn": {
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      const entries = finalizeStreamingEntries(state.entries, action.nowMs);
      if (action.note.tool_name) {
        const call: ToolCallItem = {
          id: uid("tc"),
          tool_name: action.note.tool_name,
          tool_args: action.note.tool_args,
          tool_status: action.note.tool_status,
          summary: action.note.summary,
          duration_ms: action.note.duration_ms,
          read_bytes: action.note.read_bytes,
          edit_bytes: action.note.edit_bytes,
        };
        const last = entries[entries.length - 1];
        if (last && last.kind === "tools") {
          entries[entries.length - 1] = { ...last, calls: [...last.calls, call] };
        } else {
          entries.push({ id: uid("tg"), kind: "tools", calls: [call] });
        }
      }
      return { ...state, entries };
    }

    case "checklist": {
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      const items = parseChecklistJson(action.note.items_json);
      if (items.length === 0) return state;
      const entries = [...state.entries];
      let idx = -1;
      for (let i = entries.length - 1; i >= 0; i--) {
        if (entries[i].kind === "checklist") {
          idx = i;
          break;
        }
      }
      if (idx >= 0) {
        entries[idx] = { ...entries[idx], kind: "checklist", items };
      } else {
        entries.push({ id: uid("ck"), kind: "checklist", items });
      }
      return { ...state, entries };
    }

    case "plan_ready":
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      return {
        ...state,
        entries: [...state.entries, { id: uid("pl"), kind: "plan", plan: action.note.plan }],
      };

    case "verification":
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      return {
        ...state,
        entries: [...state.entries, { id: uid("vf"), kind: "verification", data: action.note }],
      };

    case "approval_request":
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      return {
        ...state,
        entries: [
          ...finalizeStreamingEntries(state.entries, Date.now()),
          { id: uid("ap"), kind: "approval", request: action.note },
        ],
      };

    case "approval_resolved":
      return {
        ...state,
        entries: state.entries.map((entry) =>
          entry.kind === "approval" && entry.request.request_id === action.requestId
            ? { ...entry, resolution: action.resolution }
            : entry
        ),
      };

    case "perf":
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      return { ...state, perf: action.note.sample };

    case "notice":
      return {
        ...state,
        entries: [...state.entries, { id: uid("nt"), kind: "notice", data: action.note }],
      };

    case "run_end":
      if (state.activeRunId && action.note.run_id && action.note.run_id !== state.activeRunId) {
        return state;
      }
      return {
        ...state,
        running: false,
        lastRunEnd: action.note,
        entries: finalizeStreamingEntries(state.entries, action.nowMs),
      };

    case "resumed": {
      const rebuilt: FeedEntry[] = [];
      if (action.res.omitted_leading > 0) {
        rebuilt.push({ id: uid("om"), kind: "omitted", count: action.res.omitted_leading });
      }
      for (const item of action.res.transcript) {
        if (item.role === "user") {
          rebuilt.push({ id: uid("u"), kind: "user", text: item.text });
        } else if (item.role === "assistant") {
          rebuilt.push({ id: uid("a"), kind: "assistant", text: item.text, streaming: false });
        } else if (item.role === "tool") {
          const call: ToolCallItem = {
            id: uid("tc"),
            tool_name: item.tool || "tool",
            tool_args: item.args,
            tool_status: item.is_error ? "tool_error" : "ok",
            summary: item.text,
            duration_ms: 0,
            read_bytes: 0,
            edit_bytes: 0,
            truncated: item.truncated,
          };
          const prev = rebuilt[rebuilt.length - 1];
          if (prev && prev.kind === "tools") {
            rebuilt[rebuilt.length - 1] = { ...prev, calls: [...prev.calls, call] };
          } else {
            rebuilt.push({ id: uid("tg"), kind: "tools", calls: [call] });
          }
        }
      }
      return {
        ...state,
        activeRunId: action.res.run_id,
        running: false,
        perf: null,
        lastRunEnd: null,
        entries: rebuilt,
      };
    }

    case "local_error":
      return {
        ...state,
        running: false,
        entries: [
          ...finalizeStreamingEntries(state.entries, Date.now()),
          {
            id: uid("nt"),
            kind: "notice",
            data: {
              run_id: state.activeRunId ?? "",
              code: "client_error",
              severity: "error",
              subject: "Piper Desktop",
              detail: action.text,
            },
          },
        ],
      };
  }
}
