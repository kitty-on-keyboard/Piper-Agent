import React, { useState } from "react";
import {
  CheckCircle2,
  ChevronRight,
  FileEdit,
  FileText,
  FolderSearch,
  Terminal,
  Wrench,
  XCircle,
} from "lucide-react";
import { ToolCallItem } from "../lib/reducer";

interface ToolGroupProps {
  calls: ToolCallItem[];
}

function toolIcon(name: string) {
  if (name === "run_command" || name.includes("shell") || name.includes("exec")) {
    return <Terminal className="h-3.5 w-3.5" />;
  }
  if (name.includes("read") || name.includes("view_image")) {
    return <FileText className="h-3.5 w-3.5" />;
  }
  if (name.includes("write") || name.includes("replace") || name.includes("edit") || name.includes("patch")) {
    return <FileEdit className="h-3.5 w-3.5" />;
  }
  if (name.includes("search") || name.includes("list") || name.includes("locate") || name.includes("grep")) {
    return <FolderSearch className="h-3.5 w-3.5" />;
  }
  return <Wrench className="h-3.5 w-3.5" />;
}

function prettyToolName(name: string): string {
  if (name.startsWith("mcp__")) {
    const parts = name.split("__");
    if (parts.length >= 3) return `${parts[1]}:${parts.slice(2).join("_")}`;
  }
  return name.replace(/_/g, " ");
}

export const ToolGroup: React.FC<ToolGroupProps> = ({ calls }) => {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="my-2.5 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] shadow-[var(--shadow-card)]">
      <div className="divide-y divide-[var(--border)]">
        {calls.map((call) => {
          const isOpen = Boolean(expandedIds[call.id]);
          const isErr = call.tool_status !== "ok";
          return (
            <div key={call.id}>
              <button
                type="button"
                onClick={() => toggle(call.id)}
                className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-[12.5px] transition hover:bg-[var(--bg-hover)]"
              >
                <span className={isErr ? "text-[var(--err)]" : "text-[var(--ink-secondary)]"}>
                  {toolIcon(call.tool_name)}
                </span>
                <span className="font-medium text-[var(--ink)]">{prettyToolName(call.tool_name)}</span>
                {call.tool_args && (
                  <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-[var(--ink-secondary)]">
                    {call.tool_args}
                  </span>
                )}
                {!call.tool_args && <span className="flex-1" />}
                <span className="flex items-center gap-2 text-[11px] tabular-nums text-[var(--ink-muted)]">
                  {call.edit_bytes > 0 && (
                    <span className="rounded bg-[var(--bg-surface-2)] px-1.5 py-0.5">
                      ±{call.edit_bytes}B
                    </span>
                  )}
                  {call.read_bytes > 0 && (
                    <span className="rounded bg-[var(--bg-surface-2)] px-1.5 py-0.5">
                      {(call.read_bytes / 1024).toFixed(call.read_bytes >= 1024 ? 1 : 2)}KB
                    </span>
                  )}
                  {call.duration_ms > 0 && <span>{Math.round(call.duration_ms)}ms</span>}
                  {isErr ? (
                    <XCircle className="h-3.5 w-3.5 text-[var(--err)]" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-[var(--ok)]" />
                  )}
                  <ChevronRight
                    className={`h-3.5 w-3.5 transition-transform ${isOpen ? "rotate-90" : ""}`}
                  />
                </span>
              </button>
              {isOpen && (
                <div className="border-t border-[var(--border)] bg-[var(--code-bg)] px-3.5 py-2.5">
                  {call.summary ? (
                    <pre className="max-h-64 overflow-auto whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-[var(--ink-secondary)]">
                      {call.summary}
                    </pre>
                  ) : (
                    <div className="text-[11.5px] italic text-[var(--ink-muted)]">
                      No output summary recorded.
                    </div>
                  )}
                  {call.truncated && (
                    <div className="mt-1.5 text-[11px] text-[var(--warn)]">
                      Output truncated in restored session preview.
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
