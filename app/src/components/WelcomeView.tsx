import React from "react";
import { ArrowUpRight, Cpu, FolderOpen, GitBranch, Loader2, Sparkles } from "lucide-react";
import { ModelStatusNotification } from "../protocol.generated";
import { ScannedModel } from "../lib/settings";

interface WelcomeViewProps {
  workspaceRoot: string;
  branch: string;
  modelDir: string;
  modelStatus: ModelStatusNotification;
  scannedModels: ScannedModel[];
  onPickWorkspace: () => void;
  onLoadModel: () => void;
  onOpenSettings: () => void;
  onUsePrompt: (text: string) => void;
}

const STARTERS = [
  {
    title: "Map this codebase",
    prompt: "Give me a concise architectural map of this workspace and its main entry points.",
  },
  {
    title: "Review uncommitted changes",
    prompt: "Inspect the current workspace changes and flag any bugs, edge cases, or missing tests.",
  },
  {
    title: "Diagnose failing checks",
    prompt: "Find the test or build commands for this repo, run them, and summarize any failures.",
  },
];

export const WelcomeView: React.FC<WelcomeViewProps> = ({
  workspaceRoot,
  branch,
  modelDir,
  modelStatus,
  scannedModels,
  onPickWorkspace,
  onLoadModel,
  onOpenSettings,
  onUsePrompt,
}) => {
  const ready = modelStatus.state === "ready";
  const loading = modelStatus.state === "loading";
  const activeModel = scannedModels.find((m) => m.path === modelDir);

  return (
    <div className="mx-auto flex max-w-[720px] flex-col items-start justify-center px-6 py-12">
      <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-1 text-[11.5px] font-medium text-[var(--ink-secondary)] shadow-[var(--shadow-card)]">
        <Sparkles className="h-3.5 w-3.5 text-[var(--accent)]" />
        <span>Local Qwen3 · Apple Silicon MLX</span>
      </div>

      <h1 className="text-[28px] font-semibold tracking-tight text-[var(--ink)]">
        {workspaceRoot ? `Ready in ${workspaceRoot.split("/").pop()}` : "Open a workspace to begin"}
      </h1>
      <p className="mt-1.5 text-[14px] leading-relaxed text-[var(--ink-secondary)]">
        Single-process in-memory inference inside Seatbelt isolation. Your edits stay on disk with
        live git diff review — Piper never mutates git history.
      </p>

      {/* Setup readiness strip */}
      <div className="mt-6 flex w-full flex-wrap items-center gap-2.5">
        {!workspaceRoot ? (
          <button
            type="button"
            onClick={onPickWorkspace}
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--accent)] px-4 py-2 text-[13px] font-medium text-white transition hover:bg-[var(--accent-hover)]"
          >
            <FolderOpen className="h-4 w-4" />
            Choose workspace folder
          </button>
        ) : (
          <div className="inline-flex items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-1.5 text-[12px] text-[var(--ink-secondary)]">
            <GitBranch className="h-3.5 w-3.5 text-[var(--accent)]" />
            <span className="font-mono text-[var(--ink)]">{workspaceRoot}</span>
            {branch && <span className="text-[var(--ink-muted)]">({branch})</span>}
          </div>
        )}

        {!modelDir ? (
          <button
            type="button"
            onClick={onOpenSettings}
            className="inline-flex items-center gap-2 rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface)] px-3.5 py-1.5 text-[12.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-hover)]"
          >
            <Cpu className="h-3.5 w-3.5 text-[var(--accent)]" />
            Configure model checkpoint
          </button>
        ) : !ready ? (
          <button
            type="button"
            disabled={loading}
            onClick={onLoadModel}
            className="inline-flex items-center gap-2 rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface)] px-3.5 py-1.5 text-[12.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-hover)] disabled:opacity-60"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--accent)]" />
            ) : (
              <Cpu className="h-3.5 w-3.5 text-[var(--accent)]" />
            )}
            {loading
              ? "Loading Qwen3 weights…"
              : `Load ${activeModel?.name ?? modelDir.split("/").pop()}`}
          </button>
        ) : (
          <div className="inline-flex items-center gap-1.5 rounded-xl border border-[var(--ok)]/30 bg-[var(--ok-soft)] px-3 py-1.5 text-[12px] font-medium text-[var(--ok)]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--ok)]" />
            {activeModel?.name ?? modelDir.split("/").pop()} warm in memory
          </div>
        )}
      </div>

      {modelStatus.state === "failed" && modelStatus.detail && (
        <div className="mt-4 w-full rounded-xl border border-[var(--err)]/30 bg-[var(--err-soft)] px-3.5 py-2.5 text-[12.5px] text-[var(--err)]">
          {modelStatus.detail}
        </div>
      )}

      {workspaceRoot && (
        <div className="mt-8 grid w-full grid-cols-1 gap-2.5 sm:grid-cols-3">
          {STARTERS.map((item) => (
            <button
              key={item.title}
              type="button"
              onClick={() => onUsePrompt(item.prompt)}
              className="group flex flex-col justify-between rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)] p-4 text-left shadow-[var(--shadow-card)] transition hover:-translate-y-0.5 hover:border-[var(--border-strong)]"
            >
              <div>
                <div className="text-[13px] font-semibold text-[var(--ink)]">{item.title}</div>
                <div className="mt-1.5 text-[12px] leading-relaxed text-[var(--ink-secondary)]">
                  {item.prompt}
                </div>
              </div>
              <div className="mt-3 inline-flex items-center gap-1 text-[11px] font-medium text-[var(--accent)] opacity-80 group-hover:opacity-100">
                Use prompt <ArrowUpRight className="h-3 w-3" />
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
