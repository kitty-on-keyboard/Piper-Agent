import React, { useState } from "react";
import { FolderOpen, Plus, RefreshCw, ShieldAlert, Trash2, X } from "lucide-react";
import { AppSettings, formatBytes, ScannedModel, ThemePreference } from "../lib/settings";
import { bridge } from "../lib/bridge";

interface SettingsModalProps {
  open: boolean;
  settings: AppSettings;
  scannedModels: ScannedModel[];
  onClose: () => void;
  onUpdate: (next: AppSettings) => void;
  onRescanModels: () => void;
  onLoadModelNow: (modelDir: string, draftDir: string) => void;
}

type SettingsTab = "general" | "models" | "safety" | "advanced";

export const SettingsModal: React.FC<SettingsModalProps> = ({
  open,
  settings,
  scannedModels,
  onClose,
  onUpdate,
  onRescanModels,
  onLoadModelNow,
}) => {
  const [tab, setTab] = useState<SettingsTab>("general");
  const [newCmd, setNewCmd] = useState("");
  const [newScanRoot, setNewScanRoot] = useState("");

  if (!open) return null;

  const pickModelFolder = async () => {
    const res = await bridge.pickFolder("Select Qwen3 MLX Checkpoint Folder", settings.model_dir);
    if (!res.cancelled && res.path) {
      onUpdate({ ...settings, model_dir: res.path });
    }
  };

  const pickScanRoot = async () => {
    const res = await bridge.pickFolder("Add Model Library Directory");
    if (!res.cancelled && res.path) {
      if (!settings.model_scan_roots.includes(res.path)) {
        onUpdate({ ...settings, model_scan_roots: [...settings.model_scan_roots, res.path] });
      }
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-xs">
      <div className="flex h-[580px] w-full max-w-[760px] overflow-hidden rounded-2xl border border-[var(--border-strong)] bg-[var(--bg-surface)] shadow-[var(--shadow-modal)]">
        {/* Left nav */}
        <div className="w-44 shrink-0 border-r border-[var(--border)] bg-[var(--bg-sidebar)] p-3">
          <div className="px-2 pb-3 text-[13px] font-semibold text-[var(--ink)]">Settings</div>
          <nav className="space-y-0.5 text-[12.5px]">
            {(
              [
                { id: "general", label: "General" },
                { id: "models", label: "Models & MLX" },
                { id: "safety", label: "Safety & MCP" },
                { id: "advanced", label: "Sampling & Budget" },
              ] as const
            ).map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id)}
                className={`flex w-full rounded-lg px-2.5 py-1.5 text-left transition ${
                  tab === item.id
                    ? "bg-[var(--bg-active)] font-medium text-[var(--ink)]"
                    : "text-[var(--ink-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex h-12 items-center justify-between border-b border-[var(--border)] px-5">
            <span className="text-[13.5px] font-semibold text-[var(--ink)]">
              {tab === "general"
                ? "General & Appearance"
                : tab === "models"
                  ? "Local MLX Models"
                  : tab === "safety"
                    ? "Sandbox, Verification & MCP"
                    : "Sampling & Loop Limits"}
            </span>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-[var(--ink-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="flex-1 space-y-5 overflow-y-auto p-5 text-[13px]">
            {tab === "general" && (
              <>
                <div>
                  <label className="block font-medium text-[var(--ink)]">Appearance</label>
                  <p className="mt-0.5 text-[12px] text-[var(--ink-secondary)]">
                    Follow macOS Light/Dark mode automatically or pin a preference.
                  </p>
                  <div className="mt-2 inline-flex rounded-xl border border-[var(--border)] bg-[var(--bg-surface-2)] p-1 text-[12px]">
                    {(["system", "light", "dark"] as ThemePreference[]).map((t) => (
                      <button
                        key={t}
                        type="button"
                        onClick={() => onUpdate({ ...settings, theme: t })}
                        className={`rounded-lg px-3 py-1 capitalize transition ${
                          settings.theme === t
                            ? "bg-[var(--bg-surface)] font-medium text-[var(--ink)] shadow-xs"
                            : "text-[var(--ink-secondary)]"
                        }`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block font-medium text-[var(--ink)]">
                    Default Verify Contract
                  </label>
                  <p className="mt-0.5 text-[12px] text-[var(--ink-secondary)]">
                    Command run after turns that edit files (for example{" "}
                    <code className="font-mono">pytest</code> or{" "}
                    <code className="font-mono">ctest --preset gate</code>).
                  </p>
                  <input
                    type="text"
                    value={settings.verify_contract}
                    onChange={(e) => onUpdate({ ...settings, verify_contract: e.target.value })}
                    placeholder="Optional shell check command"
                    className="mt-2 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-2 font-mono text-[12.5px] text-[var(--ink)] focus:border-[var(--accent)] focus:outline-none"
                  />
                </div>
              </>
            )}

            {tab === "models" && (
              <>
                <div>
                  <div className="flex items-center justify-between">
                    <label className="font-medium text-[var(--ink)]">Active Model Folder</label>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => void pickModelFolder()}
                        className="inline-flex items-center gap-1 rounded-lg border border-[var(--border-strong)] px-2.5 py-1 text-[12px] hover:bg-[var(--bg-hover)]"
                      >
                        <FolderOpen className="h-3.5 w-3.5" />
                        Browse…
                      </button>
                      {settings.model_dir && (
                        <button
                          type="button"
                          onClick={() =>
                            onLoadModelNow(settings.model_dir, settings.draft_model_dir)
                          }
                          className="rounded-lg bg-[var(--accent)] px-2.5 py-1 text-[12px] font-medium text-white hover:bg-[var(--accent-hover)]"
                        >
                          Load now
                        </button>
                      )}
                    </div>
                  </div>
                  <input
                    type="text"
                    value={settings.model_dir}
                    onChange={(e) => onUpdate({ ...settings, model_dir: e.target.value })}
                    placeholder="/path/to/Qwen3-MLX-4bit"
                    className="mt-2 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-2 font-mono text-[12px] text-[var(--ink)] focus:outline-none"
                  />
                </div>

                <label className="flex items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--bg-canvas)] px-3.5 py-2.5">
                  <div>
                    <div className="font-medium text-[var(--ink)]">MTP Speculative Decoding</div>
                    <div className="text-[11.5px] text-[var(--ink-secondary)]">
                      Automatically pair sibling MTP draft heads with dense Qwen3 checkpoints.
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.speculative_decoding}
                    onChange={(e) =>
                      onUpdate({ ...settings, speculative_decoding: e.target.checked })
                    }
                    className="h-4 w-4 accent-[var(--accent)]"
                  />
                </label>

                <div>
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-[var(--ink)]">
                      Discovered Local Checkpoints ({scannedModels.length})
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => void pickScanRoot()}
                        className="inline-flex items-center gap-1 rounded-lg border border-[var(--border)] px-2 py-1 text-[11.5px] hover:bg-[var(--bg-hover)]"
                      >
                        <Plus className="h-3 w-3" /> Add folder
                      </button>
                      <button
                        type="button"
                        onClick={onRescanModels}
                        className="inline-flex items-center gap-1 rounded-lg border border-[var(--border)] px-2 py-1 text-[11.5px] hover:bg-[var(--bg-hover)]"
                      >
                        <RefreshCw className="h-3 w-3" /> Rescan
                      </button>
                    </div>
                  </div>

                  <div className="mt-2 space-y-1.5">
                    {scannedModels.map((m) => (
                      <button
                        key={m.path}
                        type="button"
                        onClick={() =>
                          onUpdate({
                            ...settings,
                            model_dir: m.path,
                            draft_model_dir: m.kind === "dense" ? m.mtp_path : "",
                          })
                        }
                        className={`flex w-full items-center justify-between rounded-xl border px-3 py-2 text-left transition ${
                          settings.model_dir === m.path
                            ? "border-[var(--accent)] bg-[var(--accent-soft)]/40"
                            : "border-[var(--border)] hover:bg-[var(--bg-hover)]"
                        }`}
                      >
                        <div className="min-w-0">
                          <div className="truncate font-medium text-[var(--ink)]">{m.name}</div>
                          <div className="truncate font-mono text-[11px] text-[var(--ink-muted)]">
                            {m.path}
                          </div>
                        </div>
                        <div className="shrink-0 text-right text-[11px] text-[var(--ink-secondary)]">
                          <div>{m.kind.toUpperCase()}</div>
                          {m.size_bytes > 0 && <div>{formatBytes(m.size_bytes)}</div>}
                        </div>
                      </button>
                    ))}
                    {scannedModels.length === 0 && (
                      <div className="rounded-xl border border-dashed border-[var(--border)] p-3 text-[12px] text-[var(--ink-muted)]">
                        No checkpoints found under `~/Models`, `~/models`, `~/.lmstudio/models`, or{" "}
                        <code className="font-mono">LMP_QWEN_DIR</code>. Add a library folder above.
                      </div>
                    )}
                  </div>

                  {settings.model_scan_roots.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {settings.model_scan_roots.map((r) => (
                        <span
                          key={r}
                          className="inline-flex items-center gap-1 rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] px-2 py-0.5 font-mono text-[11px]"
                        >
                          {r}
                          <button
                            type="button"
                            onClick={() =>
                              onUpdate({
                                ...settings,
                                model_scan_roots: settings.model_scan_roots.filter((x) => x !== r),
                              })
                            }
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}

                  <div className="mt-2 flex gap-2">
                    <input
                      type="text"
                      value={newScanRoot}
                      onChange={(e) => setNewScanRoot(e.target.value)}
                      placeholder="Paste extra model library path…"
                      className="flex-1 rounded-lg border border-[var(--border)] bg-[var(--bg-canvas)] px-2.5 py-1.5 font-mono text-[12px] focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const clean = newScanRoot.trim();
                        if (clean && !settings.model_scan_roots.includes(clean)) {
                          onUpdate({
                            ...settings,
                            model_scan_roots: [...settings.model_scan_roots, clean],
                          });
                          setNewScanRoot("");
                        }
                      }}
                      className="rounded-lg border border-[var(--border-strong)] px-3 py-1 text-[12px] hover:bg-[var(--bg-hover)]"
                    >
                      Add
                    </button>
                  </div>
                </div>
              </>
            )}

            {tab === "safety" && (
              <>
                <div>
                  <label className="block font-medium text-[var(--ink)]">Execution Sandbox</label>
                  <select
                    value={settings.sandbox_tier}
                    onChange={(e) =>
                      onUpdate({
                        ...settings,
                        sandbox_tier: Number(e.target.value) as 0 | 1 | 3,
                      })
                    }
                    className="mt-2 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-2 text-[12.5px] text-[var(--ink)]"
                  >
                    <option value={1}>Tier 1 — macOS Seatbelt sandbox (Recommended)</option>
                    <option value={0}>Tier 0 — Shell execution disabled</option>
                    <option value={3}>Tier 3 — Unsandboxed on host (Use with caution)</option>
                  </select>
                  {settings.sandbox_tier === 3 && (
                    <div className="mt-2 flex items-center gap-2 rounded-xl border border-[var(--warn)]/40 bg-[var(--warn-soft)] px-3 py-2 text-[12px] text-[var(--warn)]">
                      <ShieldAlert className="h-4 w-4 shrink-0" />
                      Tier 3 executes commands directly on macOS without Seatbelt isolation.
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <label className="flex items-center justify-between rounded-xl border border-[var(--border)] px-3.5 py-2">
                    <span>Auto-approve ordinary shell commands (under Seatbelt)</span>
                    <input
                      type="checkbox"
                      checked={settings.auto_approve_exec}
                      onChange={(e) =>
                        onUpdate({ ...settings, auto_approve_exec: e.target.checked })
                      }
                      className="h-4 w-4 accent-[var(--accent)]"
                    />
                  </label>
                  <label className="flex items-center justify-between rounded-xl border border-[var(--border)] px-3.5 py-2">
                    <span>Auto-approve workspace writes</span>
                    <input
                      type="checkbox"
                      checked={settings.auto_approve_writes}
                      onChange={(e) =>
                        onUpdate({ ...settings, auto_approve_writes: e.target.checked })
                      }
                      className="h-4 w-4 accent-[var(--accent)]"
                    />
                  </label>
                  <label className="flex items-center justify-between rounded-xl border border-[var(--border)] px-3.5 py-2">
                    <span>Allow workspace `.mcp.json` servers</span>
                    <input
                      type="checkbox"
                      checked={settings.allow_workspace_mcp}
                      onChange={(e) =>
                        onUpdate({ ...settings, allow_workspace_mcp: e.target.checked })
                      }
                      className="h-4 w-4 accent-[var(--accent)]"
                    />
                  </label>
                </div>

                <div>
                  <label className="block font-medium text-[var(--ink)]">
                    Always-Allowed Command Prefixes
                  </label>
                  <div className="mt-2 flex gap-2">
                    <input
                      type="text"
                      value={newCmd}
                      onChange={(e) => setNewCmd(e.target.value)}
                      placeholder="e.g. pytest or ctest --preset gate"
                      className="flex-1 rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12px]"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        const clean = newCmd.trim();
                        if (clean && !settings.allowed_commands.includes(clean)) {
                          onUpdate({
                            ...settings,
                            allowed_commands: [...settings.allowed_commands, clean],
                          });
                          setNewCmd("");
                        }
                      }}
                      className="rounded-xl border border-[var(--border-strong)] px-3 py-1.5 text-[12px] hover:bg-[var(--bg-hover)]"
                    >
                      Add
                    </button>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {settings.allowed_commands.map((cmd) => (
                      <span
                        key={cmd}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] px-2.5 py-1 font-mono text-[11.5px]"
                      >
                        {cmd}
                        <button
                          type="button"
                          onClick={() =>
                            onUpdate({
                              ...settings,
                              allowed_commands: settings.allowed_commands.filter((c) => c !== cmd),
                            })
                          }
                          className="text-[var(--ink-muted)] hover:text-[var(--err)]"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              </>
            )}

            {tab === "advanced" && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <label className="block">
                    <span className="text-[12px] text-[var(--ink-secondary)]">Temperature</span>
                    <input
                      type="number"
                      step="0.05"
                      value={settings.sampling.temperature}
                      onChange={(e) =>
                        onUpdate({
                          ...settings,
                          sampling: { ...settings.sampling, temperature: Number(e.target.value) },
                        })
                      }
                      className="mt-1 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12.5px]"
                    />
                  </label>
                  <label className="block">
                    <span className="text-[12px] text-[var(--ink-secondary)]">Top-P</span>
                    <input
                      type="number"
                      step="0.01"
                      value={settings.sampling.top_p}
                      onChange={(e) =>
                        onUpdate({
                          ...settings,
                          sampling: { ...settings.sampling, top_p: Number(e.target.value) },
                        })
                      }
                      className="mt-1 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12.5px]"
                    />
                  </label>
                  <label className="block">
                    <span className="text-[12px] text-[var(--ink-secondary)]">Context Budget</span>
                    <input
                      type="number"
                      step="4096"
                      value={settings.context_budget_tokens}
                      onChange={(e) =>
                        onUpdate({
                          ...settings,
                          context_budget_tokens: Number(e.target.value) || 96000,
                        })
                      }
                      className="mt-1 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12.5px]"
                    />
                  </label>
                  <label className="block">
                    <span className="text-[12px] text-[var(--ink-secondary)]">Max New Tokens</span>
                    <input
                      type="number"
                      step="1024"
                      value={settings.max_new_tokens}
                      onChange={(e) =>
                        onUpdate({
                          ...settings,
                          max_new_tokens: Number(e.target.value) || 32768,
                        })
                      }
                      className="mt-1 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12.5px]"
                    />
                  </label>
                  <label className="block">
                    <span className="text-[12px] text-[var(--ink-secondary)]">Max Iterations</span>
                    <input
                      type="number"
                      value={settings.max_iterations}
                      onChange={(e) =>
                        onUpdate({
                          ...settings,
                          max_iterations: Number(e.target.value) || 200,
                        })
                      }
                      className="mt-1 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12.5px]"
                    />
                  </label>
                  <label className="block">
                    <span className="text-[12px] text-[var(--ink-secondary)]">Top-K</span>
                    <input
                      type="number"
                      value={settings.sampling.top_k}
                      onChange={(e) =>
                        onUpdate({
                          ...settings,
                          sampling: { ...settings.sampling, top_k: Number(e.target.value) },
                        })
                      }
                      className="mt-1 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] px-3 py-1.5 font-mono text-[12.5px]"
                    />
                  </label>
                </div>

                <div>
                  <label className="block font-medium text-[var(--ink)]">
                    Agent Mode System Prompt Override
                  </label>
                  <textarea
                    rows={3}
                    value={settings.system_prompts.agent}
                    onChange={(e) =>
                      onUpdate({
                        ...settings,
                        system_prompts: { ...settings.system_prompts, agent: e.target.value },
                      })
                    }
                    placeholder="Leave blank to use Piper's built-in prompt"
                    className="mt-1.5 w-full rounded-xl border border-[var(--border-strong)] bg-[var(--bg-canvas)] p-2.5 text-[12px]"
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
