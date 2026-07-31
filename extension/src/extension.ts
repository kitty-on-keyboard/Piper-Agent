// Extension lifecycle (spec S12.1, S12.3, S12.4).
//
// Deliberately small. v1's extension.ts was 2,526 lines and 60% unreviewed machine
// output; the size ratchet caps every file here at 800, and the modules are split by
// responsibility: this one owns lifecycle and commands, client.ts owns the protocol,
// sidebar.ts owns the view.

import * as vscode from "vscode";
import * as path from "path";
import { SidecarClient } from "./client";
import { SidebarProvider } from "./sidebar";
import { RunSettings } from "./protocol.generated";

let client: SidecarClient | undefined;
let output: vscode.OutputChannel;
/** Set once the operator has confirmed an unsandboxed run in this window. */
let unsandboxedAcknowledged = false;

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
    max_iterations: cfg.get<number>("maxIterations", 40),
    wall_clock_seconds: cfg.get<number>("wallClockSeconds", 900),
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

export function activate(context: vscode.ExtensionContext): void {
  output = vscode.window.createOutputChannel("LM_Pipe");
  client = new SidecarClient();

  const sidebar = new SidebarProvider(context.extensionUri, client);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(SidebarProvider.viewType, sidebar),
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
    vscode.commands.registerCommand("lmPipe.start", async () => {
      const settings = settingsFromConfig();
      const errors = validate(settings);
      if (errors.length > 0) {
        void vscode.window.showErrorMessage(`LM_Pipe settings are invalid:\n${errors.join("\n")}`);
        return;
      }
      // T3 drops the filesystem jail and the egress denial. It is a legitimate choice on
      // your own machine and it is not the default, so it is confirmed once per session
      // rather than nagged about or quietly honoured.
      if (settings.sandbox_tier === 3 && !unsandboxedAcknowledged) {
        const choice = await vscode.window.showWarningMessage(
          "LM_Pipe is set to run commands UNSANDBOXED (tier 3). The agent's shell will " +
            "have your permissions: no filesystem jail, no egress denial. Wall-clock, " +
            "memory and output limits still apply.",
          { modal: true },
          "Run unsandboxed",
          "Use the sandbox instead"
        );
        if (choice === undefined) return;
        if (choice === "Use the sandbox instead") {
          settings.sandbox_tier = 1;
        } else {
          unsandboxedAcknowledged = true;
        }
      }

      const mission = await vscode.window.showInputBox({
        prompt: "Mission — the one place the deliverable is named",
        placeHolder: "e.g. Fix the failing parser test and prove it passes",
      });
      if (!mission) return;

      if (!client?.running) {
        const binary = context.asAbsolutePath(path.join("bin", "lmp_sidecar"));
        try {
          client?.start(binary);
        } catch (err) {
          void vscode.window.showErrorMessage(`Cannot start sidecar: ${String(err)}`);
          return;
        }
      }
      sidebar.beginRun(mission);
      await client?.start_run(mission, settings);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("lmPipe.cancel", async () => {
      const runId = sidebar.currentRunId;
      if (runId) await client?.cancel(runId);
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
