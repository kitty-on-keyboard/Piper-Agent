import React, { useMemo, useState } from "react";
import {
  ChevronDown,
  Cpu,
  FolderGit2,
  FolderOpen,
  GitBranch,
  History,
  Loader2,
  Plus,
  Settings,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import { ModelStatusNotification, SessionSummary } from "../protocol.generated";
import { AppSettings, formatBytes, ScannedModel } from "../lib/settings";
import { bridge } from "../lib/bridge";

interface SidebarProps {
  open: boolean;
  settings: AppSettings;
  branch: string;
  sessions: SessionSummary[];
  activeRunId: string | null;
  modelStatus: ModelStatusNotification;
  scannedModels: ScannedModel[];
  onSelectWorkspace: (path: string) => void;
  onPickWorkspace: () => void;
  onNewSession: () => void;
  onResumeSession: (runId: string) => void;
  onSelectModel: (path: string, loadImmediately?: boolean) => void;
  onUnloadModel: () => void;
  onOpenSettings: () => void;
}

function formatRelativeTime(wallNs: number): string {
  if (!wallNs || wallNs <= 0) return "";
  const ms = wallNs / 1e6;
  const diffSec = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (diffSec < 60) return "just now";
  const min = Math.floor(diffSec / 60);
  if (min < 60) return `${min}m`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h`;
  const days = Math.floor(hr / 24);
  return `${days}d`;
}

function basename(p: string): string {
  if (!p) return "";
  const clean = p.replace(/\/+$/, "");
  const idx = clean.lastIndexOf("/");
  return idx >= 0 ? clean.slice(idx + 1) : clean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  open,
  settings,
  branch,
  sessions,
  activeRunId,
  modelStatus,
  scannedModels,
  onSelectWorkspace,
  onPickWorkspace,
  onNewSession,
  onResumeSession,
  onSelectModel,
  onUnloadModel,
  onOpenSettings,
}) => {
  const [wsMenuOpen, setWsMenuOpen] = useState(false);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);

  const recentWorkspaces = useMemo(() => {
    const all = [settings.workspace_root, ...settings.recent_workspaces].filter(Boolean);
    return [...new Set(all)];
  }, [settings.workspace_root, settings.recent_workspaces]);

  if (!open) return null;

  const modelName = basename(modelStatus.model_dir || settings.model_dir) || "No model selected";
  const isReady = modelStatus.state === "ready";
  const isLoading = modelStatus.state === "loading";
  const isFailed = modelStatus.state === "failed";

  return (
    <aside className="flex h-full w-[268px] shrink-0 flex-col border-r border-[var(--border)] bg-[var(--bg-sidebar)] select-none">
      {/* Traffic-light safe drag header */}
      <div
        onMouseDown={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          bridge.windowDrag();
        }}
        className="flex h-[44px] items-center justify-between pl-[76px] pr-3"
      >
        <span className="text-[12.5px] font-semibold tracking-tight text-[var(--ink-secondary)]">
          Piper
        </span>
        <button
          type="button"
          onClick={onNewSession}
          title="New session (⌘N)"
          className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-[var(--ink-secondary)] transition hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
        >
          <Plus className="h-4 w-4" />
        </button>
      </div>

      {/* Workspace selector */}
      <div className="relative px-2.5 pt-1 pb-2">
        <button
          type="button"
          onClick={() => setWsMenuOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-2 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] px-2.5 py-2 text-left transition hover:border-[var(--border-strong)]"
        >
          <div className="flex min-w-0 items-center gap-2">
            <FolderGit2 className="h-4 w-4 shrink-0 text-[var(--accent)]" />
            <div className="min-w-0">
              <div className="truncate text-[12.5px] font-medium text-[var(--ink)]">
                {basename(settings.workspace_root) || "Choose workspace"}
              </div>
              {branch && (
                <div className="flex items-center gap-1 text-[10.5px] text-[var(--ink-muted)]">
                  <GitBranch className="h-2.5 w-2.5" />
                  <span className="truncate">{branch}</span>
                </div>
              )}
            </div>
          </div>
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-[var(--ink-muted)]" />
        </button>

        {wsMenuOpen && (
          <>
            <div className="fixed inset-0 z-30" onClick={() => setWsMenuOpen(false)} />
            <div className="absolute left-2.5 right-2.5 top-[58px] z-40 overflow-hidden rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface)] p-1 shadow-[var(--shadow-modal)]">
              <button
                type="button"
                onClick={() => {
                  setWsMenuOpen(false);
                  onPickWorkspace();
                }}
                className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[12px] font-medium text-[var(--accent)] transition hover:bg-[var(--bg-hover)]"
              >
                <FolderOpen className="h-3.5 w-3.5" />
                Open folder…
              </button>
              {recentWorkspaces.length > 0 && (
                <div className="my-1 border-t border-[var(--border)]" />
              )}
              <div className="max-h-52 overflow-y-auto">
                {recentWorkspaces.map((ws) => (
                  <button
                    key={ws}
                    type="button"
                    onClick={() => {
                      setWsMenuOpen(false);
                      onSelectWorkspace(ws);
                    }}
                    className={`flex w-full flex-col rounded-lg px-2.5 py-1.5 text-left transition hover:bg-[var(--bg-hover)] ${
                      ws === settings.workspace_root ? "bg-[var(--bg-active)]" : ""
                    }`}
                  >
                    <span className="truncate text-[12px] font-medium text-[var(--ink)]">
                      {basename(ws)}
                    </span>
                    <span className="truncate text-[10.5px] text-[var(--ink-muted)]">{ws}</span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Sessions list */}
      <div className="flex items-center justify-between px-3.5 pt-2 pb-1 text-[11px] font-medium uppercase tracking-wider text-[var(--ink-muted)]">
        <span className="inline-flex items-center gap-1.5">
          <History className="h-3 w-3" />
          Sessions
        </span>
        <span className="tabular-nums">{sessions.length}</span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {!settings.workspace_root ? (
          <div className="px-2 py-4 text-[12px] text-[var(--ink-muted)]">
            Open a workspace folder to view sessions.
          </div>
        ) : sessions.length === 0 ? (
          <div className="px-2 py-4 text-[12px] text-[var(--ink-muted)]">
            No saved sessions in this repo yet.
          </div>
        ) : (
          <div className="space-y-0.5">
            {sessions.map((s) => {
              const isActive = s.run_id === activeRunId;
              return (
                <button
                  key={s.run_id}
                  type="button"
                  onClick={() => onResumeSession(s.run_id)}
                  className={`group flex w-full flex-col gap-0.5 rounded-lg px-2.5 py-2 text-left transition ${
                    isActive
                      ? "bg-[var(--bg-active)] text-[var(--ink)]"
                      : "text-[var(--ink-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[12.5px] font-medium text-[var(--ink)]">
                      {s.mission || "Untitled session"}
                    </span>
                    <span className="shrink-0 tabular-nums text-[10.5px] text-[var(--ink-muted)]">
                      {formatRelativeTime(s.started_wall_ns)}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10.5px] text-[var(--ink-muted)]">
                    <span
                      className={`h-1.5 w-1.5 rounded-full ${
                        s.completed
                          ? "bg-[var(--ok)]"
                          : s.finished
                            ? "bg-[var(--warn)]"
                            : "bg-[var(--accent)]"
                      }`}
                    />
                    <span className="truncate">
                      {s.observations} turns · {s.termination_reason || (s.finished ? "ended" : "interrupted")}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Model status + Settings footer */}
      <div className="relative border-t border-[var(--border)] p-2.5">
        {modelMenuOpen && (
          <>
            <div className="fixed inset-0 z-30" onClick={() => setModelMenuOpen(false)} />
            <div className="absolute bottom-[88px] left-2.5 right-2.5 z-40 overflow-hidden rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface)] p-1.5 shadow-[var(--shadow-modal)]">
              <div className="flex items-center justify-between px-2 py-1 text-[11px] font-medium text-[var(--ink-muted)]">
                <span>Local MLX Checkpoints</span>
                <button
                  type="button"
                  onClick={() => {
                    setModelMenuOpen(false);
                    onOpenSettings();
                  }}
                  className="text-[var(--accent)] hover:underline"
                >
                  Manage
                </button>
              </div>
              {scannedModels.length === 0 ? (
                <div className="px-2 py-2.5 text-[11.5px] text-[var(--ink-muted)]">
                  No Qwen3 MLX checkpoints found yet. Configure folder in Settings.
                </div>
              ) : (
                <div className="max-h-52 space-y-0.5 overflow-y-auto">
                  {scannedModels.map((m) => (
                    <button
                      key={m.path}
                      type="button"
                      onClick={() => {
                        setModelMenuOpen(false);
                        onSelectModel(m.path, true);
                      }}
                      className={`flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-[var(--bg-hover)] ${
                        settings.model_dir === m.path ? "bg-[var(--bg-active)]" : ""
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="truncate text-[12px] font-medium text-[var(--ink)]">
                          {m.name}
                        </div>
                        <div className="text-[10.5px] text-[var(--ink-muted)]">
                          {m.kind.toUpperCase()}
                          {m.mtp_path ? " · MTP" : ""}
                          {m.size_bytes > 0 ? ` · ${formatBytes(m.size_bytes)}` : ""}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {isReady && (
                <>
                  <div className="my-1 border-t border-[var(--border)]" />
                  <button
                    type="button"
                    onClick={() => {
                      setModelMenuOpen(false);
                      onUnloadModel();
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-[12px] text-[var(--err)] transition hover:bg-[var(--err-soft)]"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    Unload weights from memory
                  </button>
                </>
              )}
            </div>
          </>
        )}

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setModelMenuOpen((v) => !v)}
            className="flex min-w-0 flex-1 items-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] px-2.5 py-2 text-left transition hover:border-[var(--border-strong)]"
          >
            {isLoading ? (
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-[var(--accent)]" />
            ) : (
              <Cpu
                className={`h-3.5 w-3.5 shrink-0 ${
                  isReady
                    ? "text-[var(--ok)]"
                    : isFailed
                      ? "text-[var(--err)]"
                      : "text-[var(--ink-muted)]"
                }`}
              />
            )}
            <div className="min-w-0 flex-1">
              <div className="truncate text-[12px] font-medium text-[var(--ink)]">{modelName}</div>
              <div className="truncate text-[10.5px] text-[var(--ink-muted)]">
                {isReady
                  ? `Warm${modelStatus.elapsed_ms > 0 ? ` · ${(modelStatus.elapsed_ms / 1000).toFixed(1)}s` : ""}`
                  : isLoading
                    ? "Loading weights…"
                    : isFailed
                      ? "Load failed"
                      : "Unloaded · click to load"}
              </div>
            </div>
          </button>

          <button
            type="button"
            onClick={onOpenSettings}
            title="Settings (⌘,)"
            className="inline-flex h-[42px] w-[38px] shrink-0 items-center justify-center rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] text-[var(--ink-secondary)] transition hover:border-[var(--border-strong)] hover:text-[var(--ink)]"
          >
            {settings.model_dir ? (
              <Settings className="h-4 w-4" />
            ) : (
              <SlidersHorizontal className="h-4 w-4 text-[var(--accent)]" />
            )}
          </button>
        </div>
      </div>
    </aside>
  );
};
