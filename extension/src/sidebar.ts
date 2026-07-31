// The sidebar's protocol wiring (spec S12.2).
//
// Notifications in, view messages out. What the view LOOKS like -- markup, styles, the
// typewriter, the thinking indicator -- lives in webview.ts.
//
// The transcript is one ordered feed: user turns, the answer, reasoning behind a
// disclosure (peeled off the answer body upstream by TOKEN ID, never inlined), tool rows
// where refused/denied is visually distinct from failed, verification rows carrying
// falsifiability state, and HITL cards with capability chips. The checklist and the perf
// HUD are pinned outside the feed because they are live state rather than history.

import * as vscode from "vscode";
import { SidecarClient } from "./client";
import { webviewHtml } from "./webview";
import {
  TokenNotification,
  TurnNotification,
  ChecklistNotification,
  VerificationNotification,
  ApprovalRequestNotification,
  PerfNotification,
  RunEndNotification,
} from "./protocol.generated";

export class SidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "lmPipe.sidebar";
  public currentRunId: string | undefined;
  /** True between the first notification of a run and its run_end. Decides whether typing
   *  into the box steers the live run or starts a follow-up -- the sidecar makes the same
   *  decision independently, so this only chooses the wording shown to the user. */
  private runInFlight = false;
  private view: vscode.WebviewView | undefined;
  private watcher: vscode.Disposable | undefined;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly client: SidecarClient
  ) {
    this.client.on("token", (n: TokenNotification) => this.observe(n.run_id, "token", n));
    this.client.on("turn", (n: TurnNotification) => this.observe(n.run_id, "turn", n));
    this.client.on("checklist", (n: ChecklistNotification) =>
      this.observe(n.run_id, "checklist", n)
    );
    this.client.on("verification", (n: VerificationNotification) =>
      this.observe(n.run_id, "verification", n)
    );
    this.client.on("perf", (n: PerfNotification) => this.observe(n.run_id, "perf", n));
    this.client.on("approval_request", (n: ApprovalRequestNotification) =>
      this.observe(n.run_id, "approval", n)
    );
    this.client.on("run_end", (n: RunEndNotification) => {
      // termination_reason is the one unambiguous signal for WHICH ENDING a run took
      // (S14). It is shown, not summarized away.
      this.observe(n.run_id, "run_end", n);
      this.runInFlight = false;
      this.post("idle", {});
    });
  }

  /** Every notification carries the run it belongs to, so the run id is READ off the
   *  stream rather than remembered from the start reply.
   *
   *  It was remembered from nowhere at all before this: `currentRunId` was declared,
   *  never assigned, and `lmPipe.cancel` read it -- so the Cancel command sent nothing
   *  and silently did nothing, for every run. Reading it here also keeps it correct
   *  across a follow-up, which mints a new run id without a start reply to carry it. */
  private observe(runId: string, kind: string, payload: unknown): void {
    if (runId) {
      this.currentRunId = runId;
      this.runInFlight = true;
    }
    this.post(kind, payload);
  }

  beginRun(mission: string): void {
    this.runInFlight = true;
    this.post("run_start", { mission });
  }

  private post(kind: string, payload: unknown): void {
    void this.view?.webview.postMessage({ kind, payload });
  }

  /** Keys the drawer can read and write. Everything else stays in the Settings UI --
   *  this is the set worth changing between one run and the next. */
  private static readonly LIVE_KEYS = [
    "mode",
    "sandboxTier",
    "autoApproveExec",
    "autoApproveWrites",
    "sampling.temperature",
    "sampling.topP",
    "sampling.topK",
    "sampling.minP",
    "sampling.repetitionPenalty",
    "prompts.agent",
    "prompts.plan",
    "prompts.debug",
  ];

  /** Pushes current configuration into the drawer. The drawer holds no state of its
   *  own: it renders this and writes back, so it and the Settings UI cannot drift. */
  private pushSettings(): void {
    const cfg = vscode.workspace.getConfiguration("lmPipe");
    const out: Record<string, unknown> = {};
    for (const key of SidebarProvider.LIVE_KEYS) out[key] = cfg.get(key);
    this.post("settings", out);
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [this.extensionUri] };
    view.webview.html = this.html();
    view.webview.onDidReceiveMessage(
      (msg: {
        kind: string; id?: string; approved?: boolean; text?: string;
        key?: string; value?: unknown; remember?: string;
      }) => {
        if (msg.kind === "approve" && msg.id !== undefined) {
          // "Always allow" is an approval PLUS a remembered rule. The rule is stored on
          // this side, not the sidecar's: it has to outlive the run, and the sidecar
          // deliberately owns no persistent state.
          if (msg.remember) this.remember(msg.remember);
          void this.client.approve(msg.id, msg.approved === true);
        }
        if (msg.kind === "cancel") {
          void vscode.commands.executeCommand("lmPipe.cancel");
        }
        if (msg.kind === "message" && msg.text) {
          this.send(msg.text);
        }
        if (msg.kind === "setting" && msg.key !== undefined) {
          // Only the keys the drawer owns. A webview message is untrusted input, and
          // "write whatever key it names into the user's settings" is not a thing to
          // offer on trust.
          if (!SidebarProvider.LIVE_KEYS.includes(msg.key)) return;
          void vscode.workspace
            .getConfiguration("lmPipe")
            .update(msg.key, msg.value, vscode.ConfigurationTarget.Global);
        }
      }
    );
    this.pushSettings();
    // Settings changed elsewhere -- the Settings UI, another window, a sync -- must
    // reach the drawer too, or it would show a stale copy of state it does not own.
    this.watcher?.dispose();
    this.watcher = vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("lmPipe")) this.pushSettings();
    });
  }

  /** Adds a command to the allowlist, deduplicated. */
  private remember(command: string): void {
    const trimmed = command.trim();
    if (!trimmed) return;
    const cfg = vscode.workspace.getConfiguration("lmPipe");
    const current = cfg.get<string[]>("allowedCommands", []);
    if (current.includes(trimmed)) return;
    void cfg.update(
      "allowedCommands",
      [...current, trimmed],
      vscode.ConfigurationTarget.Workspace
    );
  }

  dispose(): void {
    this.watcher?.dispose();
  }

  /** Sends the user's text and echoes it into the transcript immediately.
   *
   *  The echo is local because the sidecar does not reflect user turns back -- it puts
   *  them in the model's context, which is a different thing from putting them on the
   *  user's screen. Without it, typing into a running agent looked like typing into a
   *  void for however long the current turn had left. */
  private send(text: string): void {
    this.post("said", { text, steering: this.runInFlight });
    if (!this.runInFlight) {
      this.runInFlight = true;
    }
    void this.client.message(this.currentRunId ?? "", text);
  }

  private html(): string {
    // Markup, styles and the view script live in webview.ts. This class owns the
    // protocol wiring; mixing a stylesheet into it made both harder to read.
    return webviewHtml(Math.random().toString(36).slice(2));
  }
}
