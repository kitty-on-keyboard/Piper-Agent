// Extension lifecycle (spec S12.1, S12.3, S12.4).
//
// Split by responsibility, which is the only reason worth splitting on: this owns
// lifecycle and commands, client.ts owns the protocol, sidebar.ts owns the view, and
// webview.ts owns what the view looks like. v1's extension.ts was 2,526 lines and 60%
// unreviewed machine output -- the problem there was the unreviewed part, and four
// files of 600 unreviewed lines would have been the same product.

import * as vscode from "vscode";
import * as path from "path";
import { SidecarClient } from "./client";
import { ExtensionHost, SidebarProvider } from "./sidebar";
import { McpServerSettings, RunSettings } from "./protocol.generated";

let client: SidecarClient | undefined;
let output: vscode.OutputChannel;
/** Set once the operator has confirmed an unsandboxed run in this window. */
let unsandboxedAcknowledged = false;
/** Set once the operator has confirmed trusted MCP servers in this window. */
let trustedMcpAcknowledged = false;

function settingsFromConfig(): RunSettings {
  const cfg = vscode.workspace.getConfiguration("lmPipe");
  const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "";
  const mode = cfg.get<"plan" | "debug" | "agent">("mode", "agent");
  return {
    model_dir: cfg.get<string>("modelDir", ""),
    workspace_root: root,
    mode,
    sampling: {
      // Qwen defaults, pinned (S5.9). Changing them must move a benchmark first.
      temperature: cfg.get<number>("sampling.temperature", 0.6),
      top_p: cfg.get<number>("sampling.topP", 0.95),
      top_k: cfg.get<number>("sampling.topK", 20),
      min_p: cfg.get<number>("sampling.minP", 0.0),
      repetition_penalty: cfg.get<number>("sampling.repetitionPenalty", 1.05),
      seed: cfg.get<number>("sampling.seed", 0),
    },
    // Ceilings on a runaway, not targets (see loop::Budget). They must be read together:
    // a turn limit raised on its own just moves the cutoff to the clock and ends the run
    // for a different stated reason.
    max_iterations: cfg.get<number>("maxIterations", 200),
    wall_clock_seconds: cfg.get<number>("wallClockSeconds", 4800),
    sandbox_tier: cfg.get<number>("sandboxTier", 1),
    auto_approve_exec: cfg.get<boolean>("autoApproveExec", true),
    auto_approve_writes: cfg.get<boolean>("autoApproveWrites", true),
    require_approval: cfg.get<boolean>("requireApproval", false),
    // One prompt per mode, so switching mode switches persona. Empty is meaningful:
    // it means the built-in.
    system_prompt: cfg.get<string>(`prompts.${mode}`, ""),
    // Newline-separated on the wire: the generated protocol has no array type, and a
    // newline is the one character a shell command cannot carry unescaped.
    allowed_commands: cfg.get<string[]>("allowedCommands", []).join("\n"),
    context_budget_tokens: cfg.get<number>("contextBudgetTokens", 96000),
    max_new_tokens: cfg.get<number>("maxNewTokens", 4096),
    // The operator's check: run after any turn that writes, exit 0 = pass, output fed
    // to the model. Empty disables. Operator-owned -- the model cannot set or change it.
    verify_contract: cfg.get<string>("verifyContract", ""),
    // We can apply edits ourselves, so the sidecar routes writes back through
    // lmp/edit and VS Code's WorkspaceEdit API applies them -- undo, dirty buffers
    // and diff review all work (S12.4). A headless client leaves this false and the
    // sidecar writes directly; it must never be assumed, or an unattended run blocks
    // forever on a reply nobody will send.
    applies_edits: true,
    // Language features VS Code already hosts (workspace symbols / definition / …).
    provides_code_intel: true,
    // MCP servers connected at run start; their tools register as mcp__<name>__<tool>.
    //
    // Normalised field by field rather than passed through, because this comes from user
    // settings JSON and TypeScript's type assertion on cfg.get is a claim, not a check.
    // `trusted` in particular is compared to `true` rather than coerced: it is the
    // operator vouching that a server may run OUTSIDE the sandbox without a card for
    // every call, and any value we did not understand must not read as yes.
    mcp_servers: cfg
      .get<Partial<McpServerSettings>[]>("mcpServers", [])
      .filter((s) => typeof s?.name === "string" && typeof s?.command === "string")
      .map((s) => ({
        name: s.name as string,
        command: s.command as string,
        args: Array.isArray(s.args) ? s.args.map(String) : [],
        env: Array.isArray(s.env) ? s.env.map(String) : [],
        trusted: s.trusted === true,
      })),
  };
}

/** Every setting is validated on load and REFUSED LOUDLY if invalid. No silent
 *  fallback to a default -- that is how v1 shipped unsafe_host as the effective
 *  default (S13). */
function validate(settings: RunSettings): string[] {
  const errors: string[] = [];
  if (!settings.model_dir) errors.push("lmPipe.modelDir is required and is empty.");
  if (!settings.workspace_root) errors.push("Open a folder first: there is no workspace root.");
  if (settings.sandbox_tier < 0 || settings.sandbox_tier > 3)
    errors.push(`lmPipe.sandboxTier must be 0, 1, 2 or 3 (got ${settings.sandbox_tier}).`);
  if (settings.sampling.temperature < 0 || settings.sampling.temperature > 2)
    errors.push(`sampling.temperature must be in [0, 2] (got ${settings.sampling.temperature}).`);
  if (settings.sampling.top_p <= 0 || settings.sampling.top_p > 1)
    errors.push(`sampling.topP must be in (0, 1] (got ${settings.sampling.top_p}).`);
  if (settings.max_iterations < 1) errors.push("lmPipe.maxIterations must be at least 1.");
  if (settings.wall_clock_seconds < 1) errors.push("lmPipe.wallClockSeconds must be at least 1.");
  return errors;
}

/** The once-per-window unsandboxed confirmation.
 *
 *  T3 drops the filesystem jail and the egress denial. It is a legitimate choice on your
 *  own machine and it is not the default, so it is confirmed once per session rather than
 *  nagged about or quietly honoured. Declining downgrades the run to the sandbox rather
 *  than refusing it; dismissing the dialog cancels. */
async function confirmContainment(settings: RunSettings): Promise<boolean> {
  if (settings.sandbox_tier !== 3 || unsandboxedAcknowledged) return true;
  const choice = await vscode.window.showWarningMessage(
    "LM_Pipe is set to run commands UNSANDBOXED (tier 3). The agent's shell will " +
      "have your permissions: no filesystem jail, no egress denial. Wall-clock, " +
      "memory and output limits still apply.",
    { modal: true },
    "Run unsandboxed",
    "Use the sandbox instead"
  );
  if (choice === undefined) return false;
  if (choice === "Use the sandbox instead") {
    settings.sandbox_tier = 1;
  } else {
    unsandboxedAcknowledged = true;
  }
  return true;
}

/** Once-per-window acknowledgement for trusted MCP servers.
 *
 *  A trusted server's tools run in the SERVER's process, outside Seatbelt, and skip
 *  per-call approval cards. That is the same class of vouch as T3: legitimate on your
 *  machine, never silent. Declining clears trusted on every configured server for this
 *  run; dismissing cancels. */
async function confirmTrustedMcp(settings: RunSettings): Promise<boolean> {
  const trusted = settings.mcp_servers.filter((s) => s.trusted);
  if (trusted.length === 0 || trustedMcpAcknowledged) return true;
  const names = trusted.map((s) => s.name).join(", ");
  const choice = await vscode.window.showWarningMessage(
    `LM_Pipe will treat MCP server${trusted.length === 1 ? "" : "s"} ` +
      `'${names}' as trusted. Their tools run outside the sandbox with no per-call ` +
      `approval card — the same class of risk as unsandboxed shell (tier 3).`,
    { modal: true },
    "Trust these servers",
    "Run them untrusted"
  );
  if (choice === undefined) return false;
  if (choice === "Run them untrusted") {
    for (const s of settings.mcp_servers) s.trusted = false;
  } else {
    trustedMcpAcknowledged = true;
  }
  return true;
}

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("LM_Pipe");
  client = new SidecarClient();

  // The PROCESS, not the model. Spawning is cheap and holds nothing, so it happens
  // whenever the view needs it; the ~19 GB is a separate act the operator asks for.
  const ensureSidecar = (): string => {
    if (client?.running) return "";
    try {
      client?.start(context.asAbsolutePath(path.join("bin", "lmp_sidecar")));
      return "";
    } catch (err) {
      return `Cannot start the LM_Pipe sidecar: ${String(err)}`;
    }
  };

  const host: ExtensionHost = {
    ensureSidecar,
    settings: settingsFromConfig,
    problems: validate,
    confirmContainment,
    confirmTrustedMcp,
  };

  const sidebar = new SidebarProvider(
    context.extensionUri, client, host, context.workspaceState);
  context.subscriptions.push(
    // retainContextWhenHidden: without it the editor DESTROYS the webview every time the
    // user clicks to another panel, and rebuilds it blank when they come back -- the
    // transcript gone, the composer text gone, the model chip back to "unloaded" while
    // 19 GB of weights sit loaded in the sidecar. The transcript lives only in the view's
    // DOM (deliberately: the sidecar owns the context, the view owns the pixels), so the
    // DOM is the thing to keep. The cost is one webview's memory for the lifetime of the
    // window, which this panel -- a transcript and a settings drawer -- can afford.
    vscode.window.registerWebviewViewProvider(SidebarProvider.viewType, sidebar, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    sidebar
  );

  // Sidecar died -> surface it with the last stderr lines and offer restart (S12.3).
  client.on("exit", ({ code, stderr }: { code: number | null; stderr: string }) => {
    output.appendLine(`sidecar exited (code ${code ?? "signal"})`);
    if (stderr) output.appendLine(stderr);
    void vscode.window
      .showErrorMessage(
        `LM_Pipe sidecar exited (code ${code ?? "signal"}).`,
        "Show log",
        "Restart"
      )
      .then((choice) => {
        if (choice === "Show log") output.show(true);
        if (choice === "Restart") void vscode.commands.executeCommand("lmPipe.start");
      });
  });

  context.subscriptions.push(
    // A deliberate start over: a new mission and a clean transcript. The sidebar's
    // composer reaches the same code by a shorter road -- the first thing you type IS
    // the mission -- so this exists for the palette and for a fresh feed, not because
    // starting a run requires a dialog.
    vscode.commands.registerCommand("lmPipe.start", async () => {
      const mission = await vscode.window.showInputBox({
        prompt: "Mission — the one place the deliverable is named",
        placeHolder: "e.g. Fix the failing parser test and prove it passes",
      });
      if (!mission) return;
      await sidebar.startRun(mission);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lmPipe.cancel", async () => {
      const runId = sidebar.currentRunId;
      if (runId) await client?.cancel(runId);
    }),
    vscode.commands.registerCommand("lmPipe.loadModel", () => sidebar.loadModel()),
    vscode.commands.registerCommand("lmPipe.unloadModel", () => sidebar.unloadModel()),
    // Picks the checkpoint and remembers it. A folder picker rather than a text box
    // because lmPipe.modelDir is an absolute path to a directory of safetensors, and the
    // failure mode of typing one is a load error several steps later.
    vscode.commands.registerCommand("lmPipe.selectModel", async () => {
      const picked = await vscode.window.showOpenDialog({
        canSelectFiles: false,
        canSelectFolders: true,
        canSelectMany: false,
        openLabel: "Use this model",
        title: "Qwen3 MLX model directory (contains tokenizer.json and safetensors)",
      });
      const dir = picked?.[0]?.fsPath;
      if (!dir) return;
      await vscode.workspace
        .getConfiguration("lmPipe")
        .update("modelDir", dir, vscode.ConfigurationTarget.Global);
      // Chosen is not loaded. Asking again here would be a second click for a decision
      // the operator has already made, so the pick carries straight through to the load.
      await sidebar.loadModel(dir);
    })
  );

  context.subscriptions.push({
    dispose: () => {
      // stdin EOF is the shutdown signal; never orphan a process holding a model.
      client?.stop();
    },
  });
}

export function deactivate(): void {
  client?.stop();
}
