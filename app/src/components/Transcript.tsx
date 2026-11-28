import React, { useEffect, useRef } from "react";
import { AlertCircle, CheckCircle2, Info, ShieldX, Sparkles, XCircle } from "lucide-react";
import { ApprovalRequestNotification } from "../protocol.generated";
import { FeedEntry } from "../lib/reducer";
import { ApprovalCard } from "./ApprovalCard";
import { ChecklistCard } from "./ChecklistCard";
import { MarkdownView } from "./MarkdownView";
import { PlanCard } from "./PlanCard";
import { ThinkingBlock } from "./ThinkingBlock";
import { ToolGroup } from "./ToolGroup";

interface TranscriptProps {
  entries: FeedEntry[];
  running: boolean;
  onDecideApproval: (opts: {
    request: ApprovalRequestNotification;
    approved: boolean;
    allowWritesForRun: boolean;
    allowCommandForRun: boolean;
    persistCommand: boolean;
  }) => void;
  onExecutePlan: (planText: string) => void;
}

export const Transcript: React.FC<TranscriptProps> = ({
  entries,
  running,
  onDecideApproval,
  onExecutePlan,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: running ? "auto" : "smooth", block: "end" });
  }, [entries, running]);

  return (
    <div className="mx-auto w-full max-w-[780px] px-6 py-6">
      <div className="space-y-4">
        {entries.map((entry) => {
          switch (entry.kind) {
            case "omitted":
              return (
                <div
                  key={entry.id}
                  className="flex items-center justify-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-[11.5px] text-[var(--ink-secondary)]"
                >
                  <Info className="h-3.5 w-3.5" />
                  Showing recent transcript tail ({entry.count} earlier entries omitted from view)
                </div>
              );

            case "user":
              return (
                <div key={entry.id} className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)] px-4 py-2.5 text-[14px] leading-relaxed text-[var(--ink)] shadow-[var(--shadow-card)] whitespace-pre-wrap">
                    {entry.text}
                    {entry.images && entry.images.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {entry.images.map((p) => (
                          <span
                            key={p}
                            className="rounded-md bg-[var(--bg-surface-2)] px-2 py-0.5 font-mono text-[11px] text-[var(--ink-secondary)]"
                          >
                            {p.split("/").pop()}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );

            case "think":
              return (
                <ThinkingBlock
                  key={entry.id}
                  text={entry.text}
                  startedAtMs={entry.startedAtMs}
                  endedAtMs={entry.endedAtMs}
                />
              );

            case "assistant":
              return (
                <div key={entry.id} className="flex gap-3">
                  <div className="mt-1 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)]">
                    <Sparkles className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <MarkdownView text={entry.text} />
                    {entry.streaming && (
                      <span className="ml-1 inline-block h-3.5 w-1.5 animate-pulse rounded-sm bg-[var(--accent)] align-middle" />
                    )}
                  </div>
                </div>
              );

            case "tools":
              return <ToolGroup key={entry.id} calls={entry.calls} />;

            case "approval":
              return (
                <ApprovalCard
                  key={entry.id}
                  request={entry.request}
                  resolution={entry.resolution}
                  onDecide={onDecideApproval}
                />
              );

            case "plan":
              return (
                <PlanCard
                  key={entry.id}
                  plan={entry.plan}
                  canExecute={!running}
                  onExecuteInAgent={() => onExecutePlan(entry.plan)}
                />
              );

            case "checklist":
              return <ChecklistCard key={entry.id} items={entry.items} />;

            case "verification": {
              const v = entry.data;
              const tone = !v.ran
                ? "border-[var(--warn)]/40 bg-[var(--warn-soft)] text-[var(--warn)]"
                : v.passed
                  ? "border-[var(--ok)]/30 bg-[var(--ok-soft)] text-[var(--ok)]"
                  : "border-[var(--err)]/40 bg-[var(--err-soft)] text-[var(--err)]";
              return (
                <div key={entry.id} className={`rounded-xl border px-3.5 py-2.5 text-[12.5px] ${tone}`}>
                  <div className="flex items-center gap-2 font-medium">
                    {!v.ran ? (
                      <ShieldX className="h-4 w-4" />
                    ) : v.passed ? (
                      <CheckCircle2 className="h-4 w-4" />
                    ) : (
                      <XCircle className="h-4 w-4" />
                    )}
                    <span>
                      Verify check {!v.ran ? "did not run" : v.passed ? "passed" : "failed"}:{" "}
                      <code className="font-mono">{v.contract}</code>
                    </span>
                  </div>
                  {v.detail && (
                    <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-black/5 p-2 font-mono text-[11px] text-[var(--ink)] whitespace-pre-wrap dark:bg-black/25">
                      {v.detail}
                    </pre>
                  )}
                </div>
              );
            }

            case "notice": {
              const n = entry.data;
              const isErr = n.severity === "error";
              return (
                <div
                  key={entry.id}
                  className={`flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-[12.5px] ${
                    isErr
                      ? "border-[var(--err)]/35 bg-[var(--err-soft)] text-[var(--err)]"
                      : "border-[var(--warn)]/35 bg-[var(--warn-soft)] text-[var(--warn)]"
                  }`}
                >
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <div className="min-w-0">
                    <div className="font-medium">
                      {n.subject || n.code}
                    </div>
                    {n.detail && <div className="mt-0.5 text-[12px] opacity-90">{n.detail}</div>}
                  </div>
                </div>
              );
            }
          }
        })}
        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  );
};
