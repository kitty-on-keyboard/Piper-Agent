import React, { useMemo, useState } from "react";
import {
  Activity,
  ExternalLink,
  FileCode2,
  GitCompare,
  RefreshCw,
  RotateCcw,
  Search,
} from "lucide-react";
import { PerfSample } from "../protocol.generated";
import {
  bridge,
  FilePreviewResult,
  WorkspaceDiffFile,
  WorkspaceDiffResult,
} from "../lib/bridge";
import { filterFuzzy } from "../lib/fuzzy";

interface InspectorDrawerProps {
  open: boolean;
  workspaceRoot: string;
  diff: WorkspaceDiffResult;
  files: string[];
  perf: PerfSample | null;
  sandboxTier: 0 | 1 | 3;
  verifyContract: string;
  sidecarRunning: boolean;
  onRefreshDiff: () => void;
  onRestartSidecar: () => void;
}

type Tab = "changes" | "files" | "telemetry";

function UnifiedDiff({ diffText }: { diffText: string }) {
  const lines = diffText.split("\n");
  return (
    <pre className="overflow-x-auto font-mono text-[11.5px] leading-[1.55]">
      {lines.map((line, i) => {
        let cls = "text-[var(--ink-secondary)] px-3";
        if (line.startsWith("+") && !line.startsWith("+++ ")) {
          cls = "bg-[var(--diff-add-bg)] text-[var(--diff-add-ink)] px-3";
        } else if (line.startsWith("-") && !line.startsWith("--- ")) {
          cls = "bg-[var(--diff-del-bg)] text-[var(--diff-del-ink)] px-3";
        } else if (line.startsWith("@@")) {
          cls = "bg-[var(--accent-soft)]/50 text-[var(--accent)] px-3";
        }
        return (
          <div key={i} className={`whitespace-pre ${cls}`}>
            {line || " "}
          </div>
        );
      })}
    </pre>
  );
}

export const InspectorDrawer: React.FC<InspectorDrawerProps> = ({
  open,
  workspaceRoot,
  diff,
  files,
  perf,
  sandboxTier,
  verifyContract,
  sidecarRunning,
  onRefreshDiff,
  onRestartSidecar,
}) => {
  const [tab, setTab] = useState<Tab>("changes");
  const [selectedDiffPath, setSelectedDiffPath] = useState<string>("");
  const [fileQuery, setFileQuery] = useState("");
  const [preview, setPreview] = useState<FilePreviewResult | null>(null);

  const selectedDiff: WorkspaceDiffFile | undefined = useMemo(() => {
    if (diff.files.length === 0) return undefined;
    return diff.files.find((f) => f.path === selectedDiffPath) ?? diff.files[0];
  }, [diff.files, selectedDiffPath]);

  const filteredFiles = useMemo(() => filterFuzzy(fileQuery, files, 120), [fileQuery, files]);

  if (!open) return null;

  const openPreview = async (relPath: string) => {
    if (!workspaceRoot) return;
    const res = await bridge.readFilePreview(workspaceRoot, relPath);
    setPreview(res);
  };

  return (
    <aside className="flex h-full w-[380px] shrink-0 flex-col border-l border-[var(--border)] bg-[var(--bg-sidebar)]">
      <div className="flex h-[44px] items-center justify-between border-b border-[var(--border)] px-3">
        <div className="inline-flex rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] p-0.5 text-[11.5px]">
          <button
            type="button"
            onClick={() => setTab("changes")}
            className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 transition ${
              tab === "changes"
                ? "bg-[var(--bg-surface-2)] font-medium text-[var(--ink)]"
                : "text-[var(--ink-secondary)] hover:text-[var(--ink)]"
            }`}
          >
            <GitCompare className="h-3.5 w-3.5" />
            Changes
            {diff.files.length > 0 && (
              <span className="tabular-nums text-[10.5px] text-[var(--accent)]">
                {diff.files.length}
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={() => setTab("files")}
            className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 transition ${
              tab === "files"
                ? "bg-[var(--bg-surface-2)] font-medium text-[var(--ink)]"
                : "text-[var(--ink-secondary)] hover:text-[var(--ink)]"
            }`}
          >
            <FileCode2 className="h-3.5 w-3.5" />
            Files
          </button>
          <button
            type="button"
            onClick={() => setTab("telemetry")}
            className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 transition ${
              tab === "telemetry"
                ? "bg-[var(--bg-surface-2)] font-medium text-[var(--ink)]"
                : "text-[var(--ink-secondary)] hover:text-[var(--ink)]"
            }`}
          >
            <Activity className="h-3.5 w-3.5" />
            Engine
          </button>
        </div>

        <button
          type="button"
          onClick={onRefreshDiff}
          title="Refresh workspace inspection"
          className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-[var(--ink-secondary)] transition hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      {tab === "changes" && (
        <div className="flex min-h-0 flex-1 flex-col">
          {!workspaceRoot ? (
            <div className="p-4 text-[12.5px] text-[var(--ink-muted)]">No workspace open.</div>
          ) : !diff.is_git ? (
            <div className="p-4 text-[12.5px] text-[var(--ink-muted)]">
              This directory is not a git repository.
            </div>
          ) : diff.files.length === 0 ? (
            <div className="p-4 text-[12.5px] text-[var(--ink-muted)]">
              Working tree is clean on <span className="font-mono">{diff.branch || "HEAD"}</span>.
            </div>
          ) : (
            <>
              <div className="max-h-48 overflow-y-auto border-b border-[var(--border)] p-1.5">
                {diff.files.map((f) => {
                  const active = selectedDiff?.path === f.path;
                  return (
                    <button
                      key={f.path}
                      type="button"
                      onClick={() => setSelectedDiffPath(f.path)}
                      className={`flex w-full items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-left text-[12px] transition ${
                        active
                          ? "bg-[var(--bg-active)] text-[var(--ink)]"
                          : "text-[var(--ink-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
                      }`}
                    >
                      <span className="min-w-0 flex-1 truncate font-mono">{f.path}</span>
                      <span className="flex shrink-0 items-center gap-1.5 tabular-nums text-[11px]">
                        {f.additions > 0 && (
                          <span className="text-[var(--diff-add-ink)]">+{f.additions}</span>
                        )}
                        {f.deletions > 0 && (
                          <span className="text-[var(--diff-del-ink)]">-{f.deletions}</span>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>

              {selectedDiff && (
                <div className="flex min-h-0 flex-1 flex-col bg-[var(--code-bg)]">
                  <div className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--bg-surface)] px-3 py-1.5 text-[11.5px]">
                    <span className="truncate font-mono text-[var(--ink)]">
                      {selectedDiff.path}
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        void bridge.revealInFinder(`${workspaceRoot}/${selectedDiff.path}`)
                      }
                      title="Reveal in Finder"
                      className="inline-flex items-center gap-1 text-[11px] text-[var(--ink-secondary)] hover:text-[var(--ink)]"
                    >
                      Finder <ExternalLink className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="min-h-0 flex-1 overflow-auto py-2">
                    {selectedDiff.diff ? (
                      <UnifiedDiff diffText={selectedDiff.diff} />
                    ) : (
                      <div className="px-3 text-[12px] text-[var(--ink-muted)]">
                        No textual diff available ({selectedDiff.status}).
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {tab === "files" && (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="border-b border-[var(--border)] p-2">
            <div className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-2.5 py-1.5">
              <Search className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
              <input
                type="text"
                value={fileQuery}
                onChange={(e) => setFileQuery(e.target.value)}
                placeholder="Filter workspace files…"
                className="w-full bg-transparent text-[12.5px] text-[var(--ink)] placeholder:text-[var(--ink-muted)] focus:outline-none"
              />
            </div>
          </div>

          <div className="max-h-56 overflow-y-auto border-b border-[var(--border)] p-1.5">
            {filteredFiles.map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => void openPreview(f)}
                className={`flex w-full items-center rounded-lg px-2.5 py-1 text-left font-mono text-[11.5px] transition ${
                  preview?.path === f
                    ? "bg-[var(--bg-active)] text-[var(--ink)]"
                    : "text-[var(--ink-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
                }`}
              >
                <span className="truncate">{f}</span>
              </button>
            ))}
            {filteredFiles.length === 0 && (
              <div className="p-3 text-[12px] text-[var(--ink-muted)]">No matching files.</div>
            )}
          </div>

          <div className="min-h-0 flex-1 overflow-auto bg-[var(--code-bg)]">
            {!preview ? (
              <div className="p-4 text-[12px] text-[var(--ink-muted)]">
                Select a file above for a read-only preview.
              </div>
            ) : !preview.ok ? (
              <div className="p-4 text-[12px] text-[var(--err)]">{preview.error}</div>
            ) : preview.binary ? (
              <div className="p-4 text-[12px] text-[var(--ink-muted)]">
                Binary file ({preview.path})
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--bg-surface)] px-3 py-1.5 text-[11.5px]">
                  <span className="truncate font-mono text-[var(--ink)]">{preview.path}</span>
                  {preview.truncated && (
                    <span className="text-[10.5px] text-[var(--warn)]">Truncated 256KB</span>
                  )}
                </div>
                <pre className="p-3 font-mono text-[11.5px] leading-[1.55] text-[var(--ink)] whitespace-pre">
                  {preview.content}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      {tab === "telemetry" && (
        <div className="space-y-3 overflow-y-auto p-4 text-[12.5px]">
          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-3.5">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-semibold text-[var(--ink)]">MLX Sidecar Process</span>
              <span
                className={`inline-flex items-center gap-1.5 text-[11.5px] font-medium ${
                  sidecarRunning ? "text-[var(--ok)]" : "text-[var(--err)]"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    sidecarRunning ? "bg-[var(--ok)]" : "bg-[var(--err)]"
                  }`}
                />
                {sidecarRunning ? "Connected" : "Stopped"}
              </span>
            </div>
            <button
              type="button"
              onClick={onRestartSidecar}
              className="mt-1 inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-strong)] px-2.5 py-1 text-[11.5px] font-medium text-[var(--ink)] transition hover:bg-[var(--bg-hover)]"
            >
              <RotateCcw className="h-3 w-3" />
              Restart sidecar process
            </button>
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-3.5">
            <div className="mb-2.5 font-semibold text-[var(--ink)]">Live Inference Telemetry</div>
            {!perf ? (
              <div className="text-[12px] text-[var(--ink-muted)]">
                Metrics appear once a turn runs.
              </div>
            ) : (
              <dl className="grid grid-cols-2 gap-y-2 text-[12px]">
                <dt className="text-[var(--ink-secondary)]">TTFT</dt>
                <dd className="text-right font-mono tabular-nums text-[var(--ink)]">
                  {Math.round(perf.ttft_ms)} ms
                </dd>
                <dt className="text-[var(--ink-secondary)]">Prefill speed</dt>
                <dd className="text-right font-mono tabular-nums text-[var(--ink)]">
                  {perf.prefill_tok_per_s.toFixed(0)} tok/s
                </dd>
                <dt className="text-[var(--ink-secondary)]">Decode speed</dt>
                <dd className="text-right font-mono tabular-nums text-[var(--ink)]">
                  {perf.decode_tok_per_s.toFixed(1)} tok/s
                </dd>
                <dt className="text-[var(--ink-secondary)]">KV prefix reused</dt>
                <dd className="text-right font-mono tabular-nums text-[var(--ink)]">
                  {perf.prefill_reused_tokens.toLocaleString()} tok
                </dd>
                <dt className="text-[var(--ink-secondary)]">Context used</dt>
                <dd className="text-right font-mono tabular-nums text-[var(--ink)]">
                  {perf.context_used.toLocaleString()} / {perf.context_max.toLocaleString()}
                </dd>
                <dt className="text-[var(--ink-secondary)]">Compactions</dt>
                <dd className="text-right font-mono tabular-nums text-[var(--ink)]">
                  {perf.compactions}
                </dd>
              </dl>
            )}
          </div>

          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] p-3.5">
            <div className="mb-2 font-semibold text-[var(--ink)]">Guardrails</div>
            <div className="space-y-1.5 text-[12px] text-[var(--ink-secondary)]">
              <div>
                Sandbox:{" "}
                <span className="font-medium text-[var(--ink)]">
                  {sandboxTier === 1
                    ? "Tier 1 (macOS Seatbelt)"
                    : sandboxTier === 0
                      ? "Tier 0 (Read-only)"
                      : "Tier 3 (Unsandboxed host)"}
                </span>
              </div>
              <div>
                Verify command:{" "}
                <span className="font-mono text-[var(--ink)]">{verifyContract || "none"}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};
