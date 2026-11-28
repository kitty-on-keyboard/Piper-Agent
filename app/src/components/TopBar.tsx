import React from "react";
import { PanelLeft, PanelRight, Shield, Zap } from "lucide-react";
import { PerfSample } from "../protocol.generated";
import { bridge } from "../lib/bridge";

interface TopBarProps {
  sidebarOpen: boolean;
  inspectorOpen: boolean;
  title: string;
  running: boolean;
  perf: PerfSample | null;
  changedFilesCount: number;
  sandboxTier: 0 | 1 | 3;
  onToggleSidebar: () => void;
  onToggleInspector: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  sidebarOpen,
  inspectorOpen,
  title,
  running,
  perf,
  changedFilesCount,
  sandboxTier,
  onToggleSidebar,
  onToggleInspector,
}) => {
  const ctxPct =
    perf && perf.context_max > 0
      ? Math.min(100, Math.round((perf.context_used / perf.context_max) * 100))
      : 0;

  return (
    <header
      onMouseDown={(e) => {
        if ((e.target as HTMLElement).closest("button")) return;
        bridge.windowDrag();
      }}
      className={`flex h-[44px] shrink-0 items-center justify-between border-b border-[var(--border)] bg-[var(--bg-canvas)]/90 pr-3 backdrop-blur select-none transition-[padding] ${
        sidebarOpen ? "pl-3" : "pl-[76px]"
      }`}
    >
      <div className="flex min-w-0 items-center gap-2">
        <button
          type="button"
          onClick={onToggleSidebar}
          title="Toggle Sidebar (⌘B)"
          className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-[var(--ink-secondary)] transition hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
        >
          <PanelLeft className="h-4 w-4" />
        </button>

        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-[13px] font-medium text-[var(--ink)]">{title}</span>
          {running && (
            <span className="inline-flex items-center gap-1 rounded-full bg-[var(--accent-soft)] px-2 py-0.5 text-[11px] font-medium text-[var(--accent)]">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[var(--accent)]" />
              Running
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {perf && (
          <div className="hidden sm:flex items-center gap-2.5 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-2.5 py-1 text-[11px] tabular-nums text-[var(--ink-secondary)]">
            {perf.decode_tok_per_s > 0 && (
              <span className="inline-flex items-center gap-1 text-[var(--ink)]">
                <Zap className="h-3 w-3 text-[var(--accent)]" />
                {perf.decode_tok_per_s.toFixed(1)} tok/s
              </span>
            )}
            {perf.context_max > 0 && (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-1.5 w-12 overflow-hidden rounded-full bg-[var(--bg-surface-2)]">
                  <span
                    className="block h-full rounded-full bg-[var(--accent)]"
                    style={{ width: `${ctxPct}%` }}
                  />
                </span>
                <span>{ctxPct}% ctx</span>
              </span>
            )}
          </div>
        )}

        <span
          title={
            sandboxTier === 3
              ? "Unsandboxed host execution"
              : sandboxTier === 0
                ? "Execution disabled (Plan mode)"
                : "macOS Seatbelt sandbox active"
          }
          className={`hidden md:inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-medium ${
            sandboxTier === 3
              ? "border-[var(--warn)]/40 bg-[var(--warn-soft)] text-[var(--warn)]"
              : "border-[var(--border)] bg-[var(--bg-surface)] text-[var(--ink-secondary)]"
          }`}
        >
          <Shield className="h-3 w-3" />
          T{sandboxTier}
        </span>

        <button
          type="button"
          onClick={onToggleInspector}
          title="Toggle Inspector (⌘J)"
          className={`inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[12px] font-medium transition ${
            inspectorOpen
              ? "bg-[var(--bg-active)] text-[var(--ink)]"
              : "text-[var(--ink-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
          }`}
        >
          {changedFilesCount > 0 && (
            <span className="rounded-full bg-[var(--accent-soft)] px-1.5 py-0.2 text-[10.5px] tabular-nums text-[var(--accent)]">
              {changedFilesCount}
            </span>
          )}
          <PanelRight className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
};
