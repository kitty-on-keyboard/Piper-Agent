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
  RunSettings,
  TokenNotification,
  TurnNotification,
  ChecklistNotification,
  VerificationNotification,
  ApprovalRequestNotification,
  EditNotification,
  CodeIntelNotification,
  PerfNotification,
  ModelStatusNotification,
  PlanReadyNotification,
  RunEndNotification,
} from "./protocol.generated";
import { editPreconditionError } from "./edit_version";
import { handleCodeIntel } from "./code_intel";

/** One finished run, as the history panel shows it.
 *
 *  Deliberately NOT the transcript. Keeping every turn of every run in workspace state
 *  would be a slow leak that nobody reads; what a human wants weeks later is what they
 *  asked for, whether it worked, and roughly when.
 */
interface RunRecord {
  mission: string;
  reason: string;
  iterations: number;
  completed: boolean;
  at: number;
}

/** What the view needs from the extension host.
 *
 *  The sidebar can now start things -- which is the whole point of it, and was the whole
 *  bug: the composer's only path was `lmp/message` to a sidecar nobody had spawned, so
 *  typing into a fresh window posted into a void while the view said "Thinking". Starting
 *  needs the binary path, the configuration and the modal confirmations, all of which are
 *  the host's business and none of which belong in a view.
 *
 *  Four methods rather than a reference to the extension: what the view may ask the host
 *  to do should be readable in one place, and this is that place. */
/** The one message that has to name its own fix. Everything else the sidebar reports is
 *  something that went wrong; this is a thing the operator has simply not done yet, and
 *  it is the state a fresh install is in. */
const NO_MODEL =
  "No model is configured. Use the model button at the top of this panel to choose a " +
  "Qwen3 MLX checkpoint directory, or set lmPipe.modelDir in settings.";

export interface ExtensionHost {
  /** Spawns the sidecar if it is not already up -- the PROCESS, not the model. Cheap,
   *  idempotent, and holds no weights, so doing it on demand costs nothing. Returns the
   *  empty string on success, or the reason it could not. */
  ensureSidecar(): string;
  /** The current configuration, in the shape the sidecar wants. */
  settings(): RunSettings;
  /** Settings that are unusable, one message each. Empty means the run may proceed. */
  problems(settings: RunSettings): string[];
  /** The once-per-window unsandboxed confirmation (tier 3). May edit `settings` down to
   *  the sandbox; returns false if the operator backed out entirely. */
  confirmContainment(settings: RunSettings): Promise<boolean>;
  /** The once-per-window trusted-MCP confirmation. May clear `trusted` on every server;
   *  returns false if the operator backed out entirely. */
  confirmTrustedMcp(settings: RunSettings): Promise<boolean>;
}

export class SidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "lmPipe.sidebar";
  public currentRunId: string | undefined;
  /** True between the first notification of a run and its run_end. Decides whether typing
   *  into the box steers the live run or starts a follow-up -- the sidecar makes the same
   *  decision independently, so this only chooses the wording shown to the user. */
  private runInFlight = false;
  private view: vscode.WebviewView | undefined;
  private watcher: vscode.Disposable | undefined;
  /** The last model_status seen, replayed to a view that attaches later.
   *
   *  A webview is destroyed and rebuilt whenever the panel is hidden, so a state that
   *  lived only in the view would come back blank -- and blank has to mean something.
   *  Here it means "no model", which is also the sidecar's starting truth. */
  private model: ModelStatusNotification = {
    state: "unloaded",
    model_dir: "",
    detail: "",
    elapsed_ms: 0,
  };

  /** The mission of the run currently in flight, so run_end can file it under something
   *  a human recognises. run_end carries the outcome and not the ask. */
  private missionInFlight = "";

  /** The plan a conversational run handed over, waiting on the operator's decision.
   *
   *  Cleared when a new run starts, so an approval can never start the run that a
   *  previous, unrelated plan asked for. */
  private planReady: string | undefined;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly client: SidecarClient,
    private readonly host: ExtensionHost,
    /** Where run history lives. Workspace-scoped on purpose: "what did I ask this repo to
     *  do" is a property of the repo, and a global list would mix every project together.
     *  It survives a window reload, which is the whole point -- the transcript does not. */
    private readonly memento: vscode.Memento
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
    this.client.on("plan_ready", (n: PlanReadyNotification) => {
      // Held here as well as posted. The card's button comes back as a webview message
      // carrying nothing but "yes", because the plan is several kilobytes of markdown and
      // round-tripping it through an untrusted sender to use as a MISSION is not a thing
      // to do -- what starts the run has to be the text the sidecar sent us.
      this.planReady = n.plan;
      this.observe(n.run_id, "plan_ready", n);
    });
    this.client.on("approval_request", (n: ApprovalRequestNotification) =>
      this.observe(n.run_id, "approval", n)
    );
    this.client.on("edit", (n: EditNotification) => void this.applyEdit(n));
    this.client.on("code_intel", (n: CodeIntelNotification) =>
      void handleCodeIntel(this.client, n)
    );
    // Not run-scoped: the model outlives every run, and its state is the one thing that
    // decides whether a prompt can be answered at all.
    this.client.on("model_status", (n: ModelStatusNotification) => {
      this.model = n;
      // Unloading destroys the ContextStore in the sidecar, so the conversation this
      // side is holding a run id for no longer exists. Forgetting it here is what makes
      // the next prompt start a new run rather than a follow-up to a dead session.
      if (n.state !== "ready") {
        this.currentRunId = undefined;
        this.runInFlight = false;
      }
      this.post("model", n);
    });
    // The sidecar died. The view is mid-"Thinking" whenever this happens during a run,
    // and nothing else will ever arrive to move it off that.
    this.client.on("exit", ({ code }: { code: number | null }) => {
      this.model = { state: "unloaded", model_dir: "", detail: "", elapsed_ms: 0 };
      this.currentRunId = undefined;
      this.runInFlight = false;
      this.post("model", this.model);
      this.fail(`The sidecar exited (code ${code ?? "signal"}). Nothing is loaded.`);
    });
    this.client.on("run_end", (n: RunEndNotification) => {
      // termination_reason is the one unambiguous signal for WHICH ENDING a run took
      // (S14). It is shown, not summarized away.
      this.observe(n.run_id, "run_end", n);
      this.runInFlight = false;
      this.record(n);
      this.post("idle", {});
    });
  }

  /** One line of history per finished run.
   *
   *  `completed` means the model answered and, when an operator check is configured,
   *  that check's last reading passed -- the same claim the dot makes everywhere else
   *  in this UI.
   */
  private record(n: RunEndNotification): void {
    const runs = this.history();
    runs.unshift({
      mission: this.missionInFlight || "(untitled run)",
      reason: n.termination_reason,
      iterations: n.iterations,
      completed: n.completed,
      at: Date.now(),
    });
    // Bounded. This is a Memento, not a database, and an unbounded list in workspace
    // state is a slow leak nobody ever looks at.
    void this.memento.update("runHistory", runs.slice(0, 50));
    this.missionInFlight = "";
  }

  private history(): RunRecord[] {
    return this.memento.get<RunRecord[]>("runHistory", []);
  }

  /** Applies a workspace edit through VS Code, then answers the sidecar.
   *
   *  Through WorkspaceEdit rather than fs.writeFile so undo, dirty buffers and the diff
   *  UI all work -- that is the whole point of routing writes back here (S12.4).
   *
   *  EVERY path replies, including the failures. The sidecar blocks the run thread on
   *  this answer, so a throw that skipped the reply would wedge the run until its wall
   *  clock -- holding 19 GB of weights -- and the user would see a hang with no cause.
   */
  private async applyEdit(n: EditNotification): Promise<void> {
    try {
      const uri = vscode.Uri.file(n.path);
      const edit = new vscode.WorkspaceEdit();
      let existing: vscode.TextDocument | undefined;
      try {
        existing = await vscode.workspace.openTextDocument(uri);
      } catch {
        existing = undefined; // a new file; createFile below
      }
      // Compare against the in-memory buffer, not disk alone: a dirty human edit that
      // diverged from the sidecar's preimage must refuse rather than overwrite.
      const currentText = existing === undefined ? undefined : existing.getText();
      const precondition = editPreconditionError(
        currentText,
        n.expected_version,
        n.expected_absent
      );
      if (precondition !== undefined) {
        await this.client.editApplied(n.request_id, false, precondition);
        return;
      }
      if (existing === undefined) {
        edit.createFile(uri, { overwrite: false, ignoreIfExists: false });
        edit.insert(uri, new vscode.Position(0, 0), n.new_content);
      } else {
        const whole = new vscode.Range(
          existing.positionAt(0),
          existing.positionAt(existing.getText().length)
        );
        edit.replace(uri, whole, n.new_content);
      }
      const ok = await vscode.workspace.applyEdit(edit);
      // Saved explicitly: an applied-but-unsaved buffer means the next `shell` call --
      // a test run, a build -- reads the OLD bytes off disk, and the model would be told
      // its edit did not take. A declined save is a failed edit, never a silent success.
      if (ok) {
        const doc = await vscode.workspace.openTextDocument(uri);
        const saved = await doc.save();
        if (!saved) {
          await this.client.editApplied(
            n.request_id,
            false,
            "VS Code applied the edit but refused to save the buffer"
          );
          return;
        }
      }
      await this.client.editApplied(
        n.request_id,
        ok,
        ok ? "" : "VS Code declined to apply the edit"
      );
    } catch (err) {
      await this.client.editApplied(n.request_id, false, String(err));
    }
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

  /** `reset` wipes the transcript, and only a DELIBERATE start over does.
   *
   *  The command-palette path is one: a new mission, a clean feed. The composer path is
   *  not -- the user's message is already on screen by the time the run begins, and
   *  clearing the feed there would delete the thing they just typed. */
  beginRun(mission: string, reset = true): void {
    this.runInFlight = true;
    this.missionInFlight = mission;
    // A new run retires any plan still waiting on a decision. Left set, the Start button
    // on a card scrolled off the top of a long transcript would begin implementing a plan
    // from a conversation two missions ago.
    this.planReady = undefined;
    this.post("run_start", {
      mission,
      reset,
      // What this run is ALLOWED to do, as opposed to what the segmented control says the
      // NEXT one will be. Those are different facts, and they diverge exactly when it
      // matters -- the control is set before a run and read at lmp/start, so a mode
      // changed mid-run shows the new word over the old behaviour.
      mode: vscode.workspace.getConfiguration("lmPipe").get<string>("mode", "agent"),
    });
  }

  /** The plan handoff: approve, and the plan becomes the mission of a fresh agent run.
   *
   *  `lmp/start` rebuilds the context store by design, and that is right here -- the plan
   *  IS the handoff artifact, and the exploratory reading behind it should not be
   *  inherited. What must not be inherited is plan mode itself, so the mode is written
   *  before the settings are read rather than overridden after. */
  private async startImplementing(): Promise<void> {
    const plan = this.planReady;
    if (plan === undefined) return;
    this.planReady = undefined;

    await this.updateSetting("mode", "agent");
    if (!(await this.ready())) return;
    const settings = this.host.settings();
    const problems = this.host.problems(settings);
    if (problems.length > 0) return this.fail(problems.join("\n"));
    if (!(await this.host.confirmContainment(settings))) {
      this.fail("Cancelled: the run needs a containment choice.");
      return;
    }
    if (!(await this.host.confirmTrustedMcp(settings))) {
      this.fail("Cancelled: the run needs a trusted-MCP choice.");
      return;
    }
    // A fresh run, not a follow-up. Clearing the id is what makes the composer's next
    // message continue the IMPLEMENTATION rather than the planning conversation whose
    // session this would otherwise still be pointing at.
    this.currentRunId = undefined;
    this.beginRun(plan, false);
    const reply = await this.client.start_run(plan, settings);
    if (reply.error) this.fail(reply.error);
  }

  /** An explicit start: the mission comes from the palette, not the composer. */
  async startRun(mission: string): Promise<void> {
    if (!(await this.ready())) return;
    const settings = this.host.settings();
    const problems = this.host.problems(settings);
    if (problems.length > 0) return this.fail(problems.join("\n"));
    if (!(await this.host.confirmContainment(settings))) return;
    if (!(await this.host.confirmTrustedMcp(settings))) return;
    this.beginRun(mission);
    const reply = await this.client.start_run(mission, settings);
    if (reply.error) this.fail(reply.error);
  }

  /** Loads the weights on purpose, from the model chip.
   *
   *  Deliberately NOT called on activation. 19 GB of unified memory on a 48 GB machine
   *  is not something an editor should take because a panel became visible; the operator
   *  says when. Sending a prompt counts as saying when -- see ready(). */
  async loadModel(modelDir?: string): Promise<void> {
    const dir = modelDir ?? this.host.settings().model_dir;
    if (!dir) return this.fail(NO_MODEL);
    const err = this.host.ensureSidecar();
    if (err) return this.fail(err);
    // No await on the view: model_status carries `loading` before this settles, which is
    // the only progress there is to show for a load that owns the sidecar for a minute.
    const reply = await this.client.loadModel(dir);
    if (reply.loaded !== true) this.fail(reply.error || `could not load ${dir}`);
  }

  async unloadModel(): Promise<void> {
    if (!this.client.running) return;
    const reply = await this.client.unloadModel();
    if (reply.error) this.fail(reply.error);
  }

  /** Everything that has to be true before a prompt can be answered: a sidecar process,
   *  and weights inside it.
   *
   *  Loading here is the one place it happens implicitly, and it is not the thing the
   *  operator asked to avoid: this fires because they sent a prompt, not because the
   *  window opened. The alternative is refusing their first message to tell them to press
   *  a button that does exactly what they already asked for. */
  private async ready(): Promise<boolean> {
    const err = this.host.ensureSidecar();
    if (err) {
      this.fail(err);
      return false;
    }
    if (this.model.state === "ready") return true;
    const dir = this.host.settings().model_dir;
    if (!dir) {
      this.fail(NO_MODEL);
      return false;
    }
    const reply = await this.client.loadModel(dir);
    if (reply.loaded !== true) {
      this.fail(reply.error || `could not load ${dir}`);
      return false;
    }
    return true;
  }

  /** Says what went wrong, in the feed, and puts the view back to rest.
   *
   *  Both halves matter. Every one of these used to be a discarded promise, so the view
   *  kept whatever state it had optimistically switched to -- and "Thinking" forever is
   *  the single worst thing a surface can show, because it is indistinguishable from
   *  working. */
  private fail(text: string): void {
    this.post("error", { text });
    this.runInFlight = false;
    this.post("idle", {});
  }

  private post(kind: string, payload: unknown): void {
    void this.view?.webview.postMessage({ kind, payload });
  }

  /** Writes a setting to the scope it is ALREADY defined in, falling back to global.
   *
   *  This used to write Global unconditionally, while settingsFromConfig() reads with
   *  normal resolution -- workspace before global. So a repo carrying `lmPipe.mode` in its
   *  .vscode/settings.json silently won every time: the segmented control moved, the write
   *  landed in a scope nothing read, and the run used the workspace's mode. The control
   *  was decoration, and the failure is invisible from the UI -- you set Plan, the header
   *  says Plan, and the run writes files.
   *
   *  Whether that is what happened to any particular run cannot be established after the
   *  fact, which is the other half of the problem: nothing anywhere recorded which scope
   *  the effective value came from. */
  private async updateSetting(key: string, value: unknown): Promise<void> {
    const cfg = vscode.workspace.getConfiguration("lmPipe");
    const scope = cfg.inspect(key);
    const target =
      scope?.workspaceFolderValue !== undefined
        ? vscode.ConfigurationTarget.WorkspaceFolder
        : scope?.workspaceValue !== undefined
          ? vscode.ConfigurationTarget.Workspace
          : vscode.ConfigurationTarget.Global;
    await cfg.update(key, value, target);
  }

  /** Keys the drawer can read and write. Everything else stays in the Settings UI --
   *  this is the set worth changing between one run and the next. */
  private static readonly LIVE_KEYS = [
    "mode",
    // Not a run setting -- it never reaches the sidecar. It rides this path because the
    // drawer's settings channel is what already survives a webview rebuild, and a view
    // preference that forgets itself every time the panel is hidden is the bug, not the
    // feature.
    "showThinking",
    "sandboxTier",
    "autoApproveExec",
    "autoApproveWrites",
    // The operator's check. On the drawer because it is per-project and per-run -- the
    // command that proves THIS workspace builds is not the one that proved the last.
    "verifyContract",
    "sampling.temperature",
    "sampling.topP",
    "sampling.topK",
    "sampling.minP",
    "sampling.repetitionPenalty",
    // Both budgets, together. The turn limit is the one that keeps ending real missions
    // mid-work, and it is worthless on the drawer on its own: raised without the clock it
    // just moves the cutoff to `wall_clock` at the same wall.
    "maxIterations",
    "wallClockSeconds",
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
        key?: string; value?: unknown; remember?: string; dir?: string;
        allowWrites?: boolean;
      }) => {
        if (msg.kind === "approve" && msg.id !== undefined) {
          // "Always allow" is an approval PLUS a remembered rule. The rule is stored on
          // this side, not the sidecar's: it has to outlive the run, and the sidecar
          // deliberately owns no persistent state.
          //
          // It also has to reach the RUN, which settings cannot do: they are read at
          // lmp/start, so the run that raised this card would ask about the identical
          // command again on the next turn and the button would read as broken. The
          // sidecar latches it for the rest of the run; settings carry it past the end.
          if (msg.remember) this.remember(msg.remember);
          // "Allow writes for this run" goes the OTHER way -- to the sidecar and not to
          // settings. It must take effect on the next card of the run in flight, which
          // nothing stored here can do: settings reach the sidecar at lmp/start, one run
          // too late to help the run that asked. It is also the reason it is not
          // persisted; consent given to one mission is not consent to the next.
          void this.client.approve(
            msg.id,
            msg.approved === true,
            msg.allowWrites === true,
            msg.remember !== undefined && msg.remember !== ""
          );
        }
        if (msg.kind === "cancel") {
          void vscode.commands.executeCommand("lmPipe.cancel");
        }
        if (msg.kind === "history") {
          this.post("history", { runs: this.history() });
        }
        if (msg.kind === "message" && msg.text) {
          void this.send(msg.text);
        }
        if (msg.kind === "load_model") void this.loadModel();
        if (msg.kind === "unload_model") void this.unloadModel();
        if (msg.kind === "pick_model") void vscode.commands.executeCommand("lmPipe.selectModel");
        if (msg.kind === "ready") {
          // The view's script is up and listening -- NOW replay works. Anything posted
          // before this arrived while the webview was still parsing HTML and was
          // silently dropped, which is why the replay lives here and not below the
          // resolve: the old unconditional post raced the script load and lost often
          // enough that a loaded model showed "unloaded" after every rebuild.
          this.pushSettings();
          this.post("model", this.model);
        }
        if (msg.kind === "setting" && msg.key !== undefined) {
          // Only the keys the drawer owns. A webview message is untrusted input, and
          // "write whatever key it names into the user's settings" is not a thing to
          // offer on trust.
          if (!SidebarProvider.LIVE_KEYS.includes(msg.key)) return;
          void this.updateSetting(msg.key, msg.value);
        }
        // The plan handoff. Carries no payload on purpose -- see `planReady`.
        if (msg.kind === "start_implementing") void this.startImplementing();
        if (msg.kind === "keep_planning") this.planReady = undefined;
      }
    );
    // No replay here: it waits for the view's "ready" message. Posting now races the
    // script load -- the webview drops messages until its listener exists -- and losing
    // that race is how a loaded model showed "unloaded" after every rebuild.
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
   *  void for however long the current turn had left.
   *
   *  THREE things the text can be, and the user chooses none of them: steering for a run
   *  in flight, a follow-up over an existing conversation, or -- the case that did not
   *  exist and is why the extension appeared dead -- the mission of a brand new run. The
   *  composer is the only input this view has; it had better be able to start something.
   */
  private async send(text: string): Promise<void> {
    this.post("said", { text, steering: this.runInFlight });
    if (!(await this.ready())) return;

    // A run id survives its run: it is what makes the NEXT prompt a follow-up over the
    // same context rather than a fresh mission. It is cleared when the model unloads,
    // because that is when the context it names stops existing.
    if (this.currentRunId !== undefined) {
      this.runInFlight = true;
      const reply = await this.client.message(this.currentRunId, text);
      if (reply.error) this.fail(reply.error);
      return;
    }

    const settings = this.host.settings();
    const problems = this.host.problems(settings);
    if (problems.length > 0) return this.fail(problems.join("\n"));
    if (!(await this.host.confirmContainment(settings))) {
      this.fail("Cancelled: the run needs a containment choice.");
      return;
    }
    if (!(await this.host.confirmTrustedMcp(settings))) {
      this.fail("Cancelled: the run needs a trusted-MCP choice.");
      return;
    }
    this.beginRun(text, false);
    const reply = await this.client.start_run(text, settings);
    if (reply.error) this.fail(reply.error);
  }

  private html(): string {
    // Markup, styles and the view script live in webview.ts. This class owns the
    // protocol wiring; mixing a stylesheet into it made both harder to read.
    return webviewHtml(Math.random().toString(36).slice(2));
  }
}
