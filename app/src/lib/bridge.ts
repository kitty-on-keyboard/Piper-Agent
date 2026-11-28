import {
  ApproveParams,
  ApproveResult,
  ApprovalRequestNotification,
  CancelParams,
  CancelResult,
  ChecklistNotification,
  LoadModelParams,
  LoadModelResult,
  MessageParams,
  MessageResult,
  ModelStatusNotification,
  NoticeNotification,
  PerfNotification,
  PlanReadyNotification,
  ReadyNotification,
  ResumeParams,
  ResumeResult,
  RunEndNotification,
  SessionsParams,
  SessionsResult,
  ShutdownResult,
  StartParams,
  StartResult,
  TokenNotification,
  TurnNotification,
  UnloadModelResult,
  VerificationNotification,
} from "../protocol.generated";
import { ScannedModel } from "./settings";

export interface HostInitResult {
  app_version: string;
  sidecar_running: boolean;
  qwen_dir: string;
  home_dir: string;
  settings_path: string;
  settings: Record<string, unknown>;
}

export interface WorkspaceDiffFile {
  path: string;
  status: "modified" | "added" | "deleted" | "renamed" | "untracked";
  additions: number;
  deletions: number;
  diff: string;
}

export interface WorkspaceDiffResult {
  is_git: boolean;
  branch: string;
  files: WorkspaceDiffFile[];
  error?: string;
}

export interface WorkspaceFilesResult {
  files: string[];
  truncated: boolean;
  error?: string;
}

export interface FilePreviewResult {
  ok: boolean;
  path: string;
  content?: string;
  truncated?: boolean;
  binary?: boolean;
  error?: string;
}

export interface SaveImageResult {
  ok: boolean;
  rel_path?: string;
  error?: string;
}

export interface HostShortcutNotification {
  name: "settings" | "new_session" | "open_workspace" | "toggle_sidebar" | "toggle_inspector";
}

export interface SidecarExitNotification {
  code: number;
  stderr_tail: string[];
}

export type BridgeEvent =
  | { method: "lmp/ready"; params: ReadyNotification }
  | { method: "lmp/model_status"; params: ModelStatusNotification }
  | { method: "lmp/token"; params: TokenNotification }
  | { method: "lmp/turn"; params: TurnNotification }
  | { method: "lmp/checklist"; params: ChecklistNotification }
  | { method: "lmp/plan_ready"; params: PlanReadyNotification }
  | { method: "lmp/verification"; params: VerificationNotification }
  | { method: "lmp/approval_request"; params: ApprovalRequestNotification }
  | { method: "lmp/perf"; params: PerfNotification }
  | { method: "lmp/run_end"; params: RunEndNotification }
  | { method: "lmp/notice"; params: NoticeNotification }
  | { method: "host/shortcut"; params: HostShortcutNotification }
  | { method: "host/sidecar_exit"; params: SidecarExitNotification };

declare global {
  interface Window {
    webkit?: {
      messageHandlers?: {
        piper?: {
          postMessage: (msg: string) => void;
        };
      };
    };
    __piperReceive?: (rawJson: string) => void;
  }
}

type PendingResolver = {
  resolve: (val: unknown) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

class PiperBridge {
  private nextId = 1;
  private pending = new Map<string, PendingResolver>();
  private listeners = new Set<(event: BridgeEvent) => void>();
  private installed = false;

  get isNativeHost(): boolean {
    return typeof window !== "undefined" && Boolean(window.webkit?.messageHandlers?.piper);
  }

  install(): void {
    if (this.installed || typeof window === "undefined") return;
    this.installed = true;
    window.__piperReceive = (rawJson: string) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(rawJson) as Record<string, unknown>;
      } catch {
        return;
      }
      if (msg.id !== undefined && msg.id !== null) {
        const key = String(msg.id);
        const entry = this.pending.get(key);
        if (entry) {
          clearTimeout(entry.timer);
          this.pending.delete(key);
          if (msg.error && typeof msg.error === "object") {
            const errObj = msg.error as { message?: unknown };
            entry.reject(new Error(typeof errObj.message === "string" ? errObj.message : "RPC failed"));
          } else {
            entry.resolve(msg.result ?? {});
          }
        }
        return;
      }
      if (typeof msg.method === "string") {
        const ev = {
          method: msg.method,
          params: (msg.params ?? {}) as unknown,
        } as BridgeEvent;
        for (const cb of this.listeners) cb(ev);
      }
    };
  }

  subscribe(fn: (event: BridgeEvent) => void): () => void {
    this.install();
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  private request<T>(method: string, params: unknown = {}, timeoutMs = 120000): Promise<T> {
    this.install();
    const id = `req_${this.nextId++}`;
    const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params });

    if (!this.isNativeHost) {
      return this.browserPreviewFallback<T>(method);
    }

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timed out waiting for ${method}`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: resolve as (val: unknown) => void,
        reject,
        timer,
      });
      window.webkit!.messageHandlers!.piper!.postMessage(payload);
    });
  }

  private async browserPreviewFallback<T>(method: string): Promise<T> {
    if (method === "host/init") {
      return {
        app_version: "2.0.0-preview",
        sidecar_running: false,
        qwen_dir: "",
        home_dir: "/Users/dev",
        settings_path: "~/Library/Application Support/Piper/settings.json",
        settings: {},
      } as T;
    }
    if (method === "host/scan_models") return { models: [] } as T;
    if (method === "host/workspace_diff") return { is_git: false, branch: "", files: [] } as T;
    if (method === "host/workspace_files") return { files: [], truncated: false } as T;
    if (method === "host/pick_folder") return { path: "", cancelled: true } as T;
    if (method === "host/save_settings") return { ok: true, error: "" } as T;
    if (method === "lmp/sessions") return { sessions: [] } as T;
    if (method === "host/window_drag") return { ok: true } as T;
    throw new Error(`Open inside Piper.app to call ${method}`);
  }

  // Host methods
  hostInit(): Promise<HostInitResult> {
    return this.request("host/init", {}, 10000);
  }

  saveSettings(settings: unknown): Promise<{ ok: boolean; error: string }> {
    return this.request("host/save_settings", { settings }, 10000);
  }

  pickFolder(title: string, defaultPath = ""): Promise<{ path: string; cancelled: boolean }> {
    return this.request("host/pick_folder", { title, default_path: defaultPath }, 300000);
  }

  scanModels(roots: string[]): Promise<{ models: ScannedModel[] }> {
    return this.request("host/scan_models", { roots }, 15000);
  }

  workspaceDiff(workspaceRoot: string): Promise<WorkspaceDiffResult> {
    return this.request("host/workspace_diff", { workspace_root: workspaceRoot }, 15000);
  }

  workspaceFiles(workspaceRoot: string, limit = 2000): Promise<WorkspaceFilesResult> {
    return this.request("host/workspace_files", { workspace_root: workspaceRoot, limit }, 15000);
  }

  readFilePreview(workspaceRoot: string, path: string): Promise<FilePreviewResult> {
    return this.request("host/read_file_preview", { workspace_root: workspaceRoot, path }, 15000);
  }

  saveImageAttachment(
    workspaceRoot: string,
    ext: string,
    base64Data: string
  ): Promise<SaveImageResult> {
    return this.request(
      "host/save_image_attachment",
      { workspace_root: workspaceRoot, ext, base64_data: base64Data },
      15000
    );
  }

  revealInFinder(path: string): Promise<{ ok: boolean }> {
    return this.request("host/reveal_in_finder", { path }, 5000);
  }

  openExternal(url: string): Promise<{ ok: boolean }> {
    return this.request("host/open_external", { url }, 5000);
  }

  restartSidecar(): Promise<{ ok: boolean; error: string }> {
    return this.request("host/restart_sidecar", {}, 15000);
  }

  windowDrag(): void {
    if (!this.isNativeHost) return;
    window.webkit!.messageHandlers!.piper!.postMessage(
      JSON.stringify({ jsonrpc: "2.0", id: null, method: "host/window_drag", params: {} })
    );
  }

  // Sidecar lmp/* methods
  loadModel(params: LoadModelParams): Promise<LoadModelResult> {
    return this.request("lmp/load_model", params, 600000);
  }

  unloadModel(): Promise<UnloadModelResult> {
    return this.request("lmp/unload_model", {}, 30000);
  }

  startRun(params: StartParams): Promise<StartResult> {
    return this.request("lmp/start", params, 60000);
  }

  sendMessage(params: MessageParams): Promise<MessageResult> {
    return this.request("lmp/message", params, 60000);
  }

  cancelRun(params: CancelParams): Promise<CancelResult> {
    return this.request("lmp/cancel", params, 10000);
  }

  approveTool(params: ApproveParams): Promise<ApproveResult> {
    return this.request("lmp/approve", params, 15000);
  }

  listSessions(params: SessionsParams): Promise<SessionsResult> {
    return this.request("lmp/sessions", params, 15000);
  }

  resumeSession(params: ResumeParams): Promise<ResumeResult> {
    return this.request("lmp/resume", params, 30000);
  }

  shutdownSidecar(): Promise<ShutdownResult> {
    return this.request("lmp/shutdown", {}, 10000);
  }
}

export const bridge = new PiperBridge();
