import { McpServerSettings, Mode, RunSettings, SamplingSettings } from "../protocol.generated";

export type ThemePreference = "system" | "light" | "dark";

export interface ScannedModel {
  path: string;
  name: string;
  kind: "dense" | "moe" | "mtp" | "unknown";
  mtp_path: string;
  size_bytes: number;
}

export interface AppSettings {
  theme: ThemePreference;
  workspace_root: string;
  recent_workspaces: string[];
  model_dir: string;
  draft_model_dir: string;
  speculative_decoding: boolean;
  model_scan_roots: string[];
  mode: Mode;
  reasoning_effort: "low" | "medium" | "xhigh" | "";
  sandbox_tier: 0 | 1 | 3;
  auto_approve_exec: boolean;
  auto_approve_writes: boolean;
  require_approval: boolean;
  commit_think: boolean;
  shadow_compact: boolean;
  context_budget_tokens: number;
  max_new_tokens: number;
  max_iterations: number;
  wall_clock_seconds: number;
  stall_seconds: number;
  sampling: SamplingSettings;
  allowed_commands: string[];
  verify_contract: string;
  system_prompts: Record<Mode, string>;
  mcp_servers: McpServerSettings[];
  allow_workspace_mcp: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  theme: "system",
  workspace_root: "",
  recent_workspaces: [],
  model_dir: "",
  draft_model_dir: "",
  speculative_decoding: true,
  model_scan_roots: [],
  mode: "agent",
  reasoning_effort: "medium",
  sandbox_tier: 1,
  auto_approve_exec: true,
  auto_approve_writes: true,
  require_approval: false,
  commit_think: true,
  shadow_compact: true,
  context_budget_tokens: 96000,
  max_new_tokens: 32768,
  max_iterations: 200,
  wall_clock_seconds: 14400,
  stall_seconds: 1200,
  sampling: {
    temperature: 0.6,
    top_p: 0.95,
    top_k: 20,
    min_p: 0.0,
    repetition_penalty: 1.05,
    seed: 0,
  },
  allowed_commands: [],
  verify_contract: "",
  system_prompts: {
    plan: "",
    debug: "",
    agent: "",
  },
  mcp_servers: [],
  allow_workspace_mcp: false,
};

/** Rejects empty entries, multiline injections, and shell chaining operators so allowlist rules stay clean. */
export function sanitizeAllowedCommands(entries: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of entries) {
    const trimmed = raw.trim();
    if (!trimmed) continue;
    if (/[\r\n\0]/.test(trimmed)) continue;
    if (seen.has(trimmed)) continue;
    seen.add(trimmed);
    out.push(trimmed);
  }
  return out;
}

export function mergeLoadedSettings(raw: unknown, fallbackModelDir = ""): AppSettings {
  const obj = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const modeVal = obj.mode;
  const mode: Mode = modeVal === "plan" || modeVal === "debug" || modeVal === "agent" ? modeVal : "agent";
  const themeVal = obj.theme;
  const theme: ThemePreference =
    themeVal === "light" || themeVal === "dark" || themeVal === "system" ? themeVal : "system";
  const effortVal = obj.reasoning_effort;
  const reasoning_effort =
    effortVal === "low" || effortVal === "medium" || effortVal === "xhigh" || effortVal === ""
      ? effortVal
      : "medium";
  const tierRaw = Number(obj.sandbox_tier);
  const sandbox_tier: 0 | 1 | 3 = tierRaw === 0 ? 0 : tierRaw === 3 ? 3 : 1;

  const rawSampling =
    obj.sampling && typeof obj.sampling === "object" ? (obj.sampling as Record<string, unknown>) : {};
  const num = (v: unknown, def: number, min: number, max: number) => {
    const n = typeof v === "number" && Number.isFinite(v) ? v : def;
    return Math.min(max, Math.max(min, n));
  };
  const str = (v: unknown, def = "") => (typeof v === "string" ? v : def);
  const bool = (v: unknown, def: boolean) => (typeof v === "boolean" ? v : def);
  const strArr = (v: unknown) =>
    Array.isArray(v) ? v.filter((x): x is string => typeof x === "string" && x.trim().length > 0) : [];

  const rawPrompts =
    obj.system_prompts && typeof obj.system_prompts === "object"
      ? (obj.system_prompts as Record<string, unknown>)
      : {};

  const rawMcp = Array.isArray(obj.mcp_servers) ? obj.mcp_servers : [];
  const mcp_servers: McpServerSettings[] = rawMcp
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
    .map((item) => ({
      name: str(item.name).trim(),
      command: str(item.command).trim(),
      args: strArr(item.args),
      env: strArr(item.env),
      trusted: item.trusted === true,
    }))
    .filter((s) => s.name.length > 0 && s.command.length > 0);

  return {
    theme,
    workspace_root: str(obj.workspace_root),
    recent_workspaces: strArr(obj.recent_workspaces).slice(0, 12),
    model_dir: str(obj.model_dir) || fallbackModelDir,
    draft_model_dir: str(obj.draft_model_dir),
    speculative_decoding: bool(obj.speculative_decoding, true),
    model_scan_roots: strArr(obj.model_scan_roots),
    mode,
    reasoning_effort,
    sandbox_tier,
    auto_approve_exec: bool(obj.auto_approve_exec, true),
    auto_approve_writes: bool(obj.auto_approve_writes, true),
    require_approval: bool(obj.require_approval, false),
    commit_think: bool(obj.commit_think, true),
    shadow_compact: bool(obj.shadow_compact, true),
    context_budget_tokens: Math.round(num(obj.context_budget_tokens, 96000, 2048, 262144)),
    max_new_tokens: Math.round(num(obj.max_new_tokens, 32768, 256, 131072)),
    max_iterations: Math.round(num(obj.max_iterations, 200, 1, 2000)),
    wall_clock_seconds: Math.round(num(obj.wall_clock_seconds, 14400, 30, 86400)),
    stall_seconds: Math.round(num(obj.stall_seconds, 1200, 30, 14400)),
    sampling: {
      temperature: num(rawSampling.temperature, 0.6, 0, 2),
      top_p: num(rawSampling.top_p, 0.95, 0.01, 1),
      top_k: Math.round(num(rawSampling.top_k, 20, 0, 500)),
      min_p: num(rawSampling.min_p, 0.0, 0, 1),
      repetition_penalty: num(rawSampling.repetition_penalty, 1.05, 0.5, 2),
      seed: Math.round(num(rawSampling.seed, 0, 0, 2147483647)),
    },
    allowed_commands: sanitizeAllowedCommands(strArr(obj.allowed_commands)),
    verify_contract: str(obj.verify_contract),
    system_prompts: {
      plan: str(rawPrompts.plan),
      debug: str(rawPrompts.debug),
      agent: str(rawPrompts.agent),
    },
    mcp_servers,
    allow_workspace_mcp: bool(obj.allow_workspace_mcp, false),
  };
}

export function resolveDraftModelDir(
  settings: AppSettings,
  scannedModels: ScannedModel[]
): string {
  if (!settings.speculative_decoding || !settings.model_dir) return "";
  const match = scannedModels.find((m) => m.path === settings.model_dir);
  if (match) {
    if (match.kind !== "dense") return "";
    return settings.draft_model_dir || match.mtp_path || "";
  }
  return settings.draft_model_dir;
}

export function toRunSettings(
  settings: AppSettings,
  scannedModels: ScannedModel[] = []
): RunSettings {
  return {
    model_dir: settings.model_dir,
    draft_model_dir: resolveDraftModelDir(settings, scannedModels),
    workspace_root: settings.workspace_root,
    mode: settings.mode,
    sampling: { ...settings.sampling },
    max_iterations: settings.max_iterations,
    wall_clock_seconds: settings.wall_clock_seconds,
    stall_seconds: settings.stall_seconds,
    sandbox_tier: settings.mode === "plan" ? 0 : settings.sandbox_tier,
    auto_approve_exec: settings.auto_approve_exec,
    auto_approve_writes: settings.auto_approve_writes,
    require_approval: settings.require_approval,
    // Piper Desktop lets lmp_sidecar write directly to disk safely inside Seatbelt.
    applies_edits: false,
    provides_code_intel: false,
    system_prompt: settings.system_prompts[settings.mode] ?? "",
    allowed_commands: sanitizeAllowedCommands(settings.allowed_commands).join("\n"),
    context_budget_tokens: settings.context_budget_tokens,
    max_new_tokens: settings.max_new_tokens,
    reasoning_effort: settings.reasoning_effort,
    verify_contract: settings.verify_contract,
    mcp_servers: settings.mcp_servers,
    allow_workspace_mcp: settings.allow_workspace_mcp,
    commit_think: settings.commit_think,
    shadow_compact: settings.shadow_compact,
  };
}

export function pushRecentWorkspace(prev: string[], dir: string, cap = 8): string[] {
  const clean = dir.trim();
  if (!clean) return prev;
  return [clean, ...prev.filter((p) => p !== clean)].slice(0, cap);
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) return `${mb.toFixed(0)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}
