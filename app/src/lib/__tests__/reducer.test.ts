import { describe, expect, it } from "vitest";
import { createInitialSessionState, sessionReducer } from "../reducer";

describe("sessionReducer", () => {
  it("streams think and assistant tokens separately and closes think timer", () => {
    let s = createInitialSessionState();
    s = sessionReducer(s, { type: "user_start", runId: "r1", text: "Fix bug" });
    s = sessionReducer(s, {
      type: "token",
      note: { run_id: "r1", text: "Checking...", channel: "think" },
      nowMs: 1000,
    });
    s = sessionReducer(s, {
      type: "token",
      note: { run_id: "r1", text: " done.", channel: "think" },
      nowMs: 1250,
    });
    s = sessionReducer(s, {
      type: "token",
      note: { run_id: "r1", text: "Here is the fix.", channel: "text" },
      nowMs: 1500,
    });

    expect(s.entries).toHaveLength(3);
    expect(s.entries[1]).toMatchObject({
      kind: "think",
      text: "Checking... done.",
      startedAtMs: 1000,
      endedAtMs: 1500,
    });
    expect(s.entries[2]).toMatchObject({
      kind: "assistant",
      text: "Here is the fix.",
      streaming: true,
    });

    s = sessionReducer(s, {
      type: "run_end",
      note: {
        run_id: "r1",
        termination_reason: "completed",
        iterations: 2,
        completed: true,
        unfinished_items: 0,
      },
      nowMs: 2000,
    });
    expect(s.running).toBe(false);
    expect(s.entries[2]).toMatchObject({ kind: "assistant", streaming: false });
  });

  it("groups consecutive tool turns and resolves approval requests", () => {
    let s = createInitialSessionState();
    s = sessionReducer(s, { type: "user_start", runId: "r1", text: "Inspect" });
    s = sessionReducer(s, {
      type: "turn",
      note: {
        run_id: "r1",
        outcome: "tool_call_executed",
        tool_name: "read_file",
        tool_args: "src/main.cpp",
        tool_status: "ok",
        summary: "Read 12 lines",
        duration_ms: 4.2,
        think_tokens: 10,
        text_tokens: 0,
        tool_tokens: 12,
        batch_index: 0,
        batch_count: 2,
        read_bytes: 300,
        edit_bytes: 0,
        cap_phase: "",
      },
      nowMs: 100,
    });
    s = sessionReducer(s, {
      type: "turn",
      note: {
        run_id: "r1",
        outcome: "tool_call_executed",
        tool_name: "list_dir",
        tool_args: "src",
        tool_status: "ok",
        summary: "Listed 5 entries",
        duration_ms: 1.5,
        think_tokens: 0,
        text_tokens: 0,
        tool_tokens: 8,
        batch_index: 1,
        batch_count: 2,
        read_bytes: 0,
        edit_bytes: 0,
        cap_phase: "",
      },
      nowMs: 110,
    });

    expect(s.entries).toHaveLength(2);
    const group = s.entries[1];
    expect(group.kind).toBe("tools");
    if (group.kind === "tools") {
      expect(group.calls).toHaveLength(2);
    }

    s = sessionReducer(s, {
      type: "approval_request",
      note: {
        request_id: "req_1",
        run_id: "r1",
        tool_name: "run_command",
        preview: "rm -rf build",
        command: "rm -rf build",
        irreversible: true,
        can_remember: false,
        risk: 0.9,
        capabilities: {
          writes_outside_workspace: false,
          reads_outside_workspace: false,
          destroys_data: true,
          rewrites_vcs_history: false,
          network_access: false,
          spawns_unbounded_process: false,
          signals_foreign_process: false,
          escalates_privileges: false,
          parse_status: "full",
        },
      },
    });
    s = sessionReducer(s, { type: "approval_resolved", requestId: "req_1", resolution: "denied" });
    const ap = s.entries.find((e) => e.kind === "approval");
    expect(ap && ap.kind === "approval" ? ap.resolution : undefined).toBe("denied");
  });

  it("hydrates resumed transcripts including omitted_leading marker", () => {
    let s = createInitialSessionState();
    s = sessionReducer(s, {
      type: "resumed",
      res: {
        ok: true,
        why: "",
        run_id: "run_99",
        mission: "Refactor",
        turns_restored: 4,
        omitted_leading: 3,
        transcript: [
          { role: "user", text: "Do thing", tool: "", args: "", is_error: false, truncated: false },
          { role: "tool", text: "ok", tool: "read_file", args: "a.ts", is_error: false, truncated: true },
          { role: "assistant", text: "Done.", tool: "", args: "", is_error: false, truncated: false },
        ],
      },
    });

    expect(s.activeRunId).toBe("run_99");
    expect(s.entries[0]).toMatchObject({ kind: "omitted", count: 3 });
    expect(s.entries[1]).toMatchObject({ kind: "user", text: "Do thing" });
    expect(s.entries[2]).toMatchObject({ kind: "tools" });
    expect(s.entries[3]).toMatchObject({ kind: "assistant", text: "Done.", streaming: false });
  });
});
