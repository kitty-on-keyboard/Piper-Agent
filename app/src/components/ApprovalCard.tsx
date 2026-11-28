import React from "react";
import { AlertTriangle, Check, ShieldAlert, ShieldCheck, X } from "lucide-react";
import { ApprovalRequestNotification } from "../protocol.generated";
import { ApprovalResolution } from "../lib/reducer";

interface ApprovalCardProps {
  request: ApprovalRequestNotification;
  resolution?: ApprovalResolution;
  onDecide: (opts: {
    request: ApprovalRequestNotification;
    approved: boolean;
    allowWritesForRun: boolean;
    allowCommandForRun: boolean;
    persistCommand: boolean;
  }) => void;
}

const CAPABILITY_LABELS: Array<{ key: keyof ApprovalRequestNotification["capabilities"]; label: string }> = [
  { key: "destroys_data", label: "Destroys data" },
  { key: "writes_outside_workspace", label: "Writes outside repo" },
  { key: "reads_outside_workspace", label: "Reads outside repo" },
  { key: "rewrites_vcs_history", label: "Rewrites git history" },
  { key: "network_access", label: "Network access" },
  { key: "spawns_unbounded_process", label: "Unbounded process" },
  { key: "signals_foreign_process", label: "Signals foreign process" },
  { key: "escalates_privileges", label: "Privilege escalation" },
];

export const ApprovalCard: React.FC<ApprovalCardProps> = ({
  request,
  resolution,
  onDecide,
}) => {
  const chips = CAPABILITY_LABELS.filter(
    (c) => request.capabilities[c.key] === true
  ).map((c) => c.label);

  const isWriteTool =
    request.tool_name === "write_file" ||
    request.tool_name === "replace_in_file" ||
    request.tool_name === "apply_patch";

  return (
    <div
      className={`my-3 rounded-2xl border p-4 shadow-[var(--shadow-card)] ${
        request.irreversible
          ? "border-[var(--err)]/40 bg-[var(--err-soft)]/50"
          : "border-[var(--warn)]/40 bg-[var(--bg-surface)]"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          {request.irreversible ? (
            <ShieldAlert className="h-4.5 w-4.5 text-[var(--err)]" />
          ) : (
            <AlertTriangle className="h-4.5 w-4.5 text-[var(--warn)]" />
          )}
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[13px] font-semibold text-[var(--ink)]">
                Approval required · {request.tool_name}
              </span>
              {request.irreversible && (
                <span className="rounded-full bg-[var(--err-soft)] px-2 py-0.5 text-[10.5px] font-medium text-[var(--err)]">
                  Irreversible
                </span>
              )}
            </div>
            <div className="text-[11.5px] text-[var(--ink-secondary)]">
              Risk score {request.risk.toFixed(2)} · parse {request.capabilities.parse_status || "unknown"}
            </div>
          </div>
        </div>

        {resolution && (
          <span
            className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11.5px] font-medium ${
              resolution === "approved"
                ? "bg-[var(--ok-soft)] text-[var(--ok)]"
                : "bg-[var(--err-soft)] text-[var(--err)]"
            }`}
          >
            {resolution === "approved" ? (
              <ShieldCheck className="h-3.5 w-3.5" />
            ) : (
              <X className="h-3.5 w-3.5" />
            )}
            {resolution === "approved" ? "Approved" : "Denied"}
          </span>
        )}
      </div>

      <pre className="mt-3 max-h-48 overflow-auto rounded-xl border border-[var(--border)] bg-[var(--code-bg)] p-3 font-mono text-[12px] text-[var(--ink)] whitespace-pre-wrap">
        {request.command || request.preview}
      </pre>

      {chips.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {chips.map((label) => (
            <span
              key={label}
              className="rounded-md border border-[var(--border)] bg-[var(--bg-surface-2)] px-2 py-0.5 text-[11px] font-medium text-[var(--ink-secondary)]"
            >
              {label}
            </span>
          ))}
        </div>
      )}

      {!resolution && (
        <div className="mt-3.5 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() =>
              onDecide({
                request,
                approved: false,
                allowWritesForRun: false,
                allowCommandForRun: false,
                persistCommand: false,
              })
            }
            className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface)] px-3 py-1.5 text-[12.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-hover)]"
          >
            <X className="h-3.5 w-3.5" />
            Deny
          </button>

          <button
            type="button"
            onClick={() =>
              onDecide({
                request,
                approved: true,
                allowWritesForRun: false,
                allowCommandForRun: false,
                persistCommand: false,
              })
            }
            className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--accent)] px-3.5 py-1.5 text-[12.5px] font-medium text-white transition hover:bg-[var(--accent-hover)]"
          >
            <Check className="h-3.5 w-3.5" />
            Allow once
          </button>

          {!request.irreversible && request.command && request.can_remember && (
            <button
              type="button"
              onClick={() =>
                onDecide({
                  request,
                  approved: true,
                  allowWritesForRun: false,
                  allowCommandForRun: true,
                  persistCommand: true,
                })
              }
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-[12.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-active)]"
            >
              Always allow
            </button>
          )}

          {!request.irreversible && request.command && !request.can_remember && (
            <button
              type="button"
              onClick={() =>
                onDecide({
                  request,
                  approved: true,
                  allowWritesForRun: false,
                  allowCommandForRun: true,
                  persistCommand: false,
                })
              }
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-[12.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-active)]"
            >
              Allow for this run
            </button>
          )}

          {!request.irreversible && isWriteTool && (
            <button
              type="button"
              onClick={() =>
                onDecide({
                  request,
                  approved: true,
                  allowWritesForRun: true,
                  allowCommandForRun: false,
                  persistCommand: false,
                })
              }
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-[12.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-active)]"
            >
              Allow writes this run
            </button>
          )}
        </div>
      )}
    </div>
  );
};
