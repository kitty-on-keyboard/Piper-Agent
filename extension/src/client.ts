// Protocol client: spawns the sidecar and speaks lmp/* over stdio (spec S4, S12.3).
//
// Framing is symmetric with the C++ side (src/surface/transport.cpp): newline-delimited
// JSON, and the READER assembles whole messages before anything inspects them. A read
// boundary is never semantically visible on either end.

import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import {
  PROTOCOL_VERSION,
  RunSettings,
  StartParams,
  MessageResult,
  LoadModelResult,
  RunEndNotification,
  TurnNotification,
  TokenNotification,
  ChecklistNotification,
  ApprovalRequestNotification,
  EditNotification,
  VerificationNotification,
  PerfNotification,
  ModelStatusNotification,
} from "./protocol.generated";

export interface SidecarEvents {
  token: TokenNotification;
  turn: TurnNotification;
  checklist: ChecklistNotification;
  verification: VerificationNotification;
  approval_request: ApprovalRequestNotification;
  edit: EditNotification;
  perf: PerfNotification;
  model_status: ModelStatusNotification;
  run_end: RunEndNotification;
  exit: { code: number | null; stderr: string };
}

/** What a request settles to: the result, or a STATED failure. Never a rejection.
 *
 *  It used to be a rejection, and every call site in sidebar.ts discarded it with `void`.
 *  So "the sidecar is not running" -- which is the state the extension is in until
 *  something starts one -- became an unhandled promise nobody saw, while the view had
 *  already switched itself to "Thinking". Typing into the sidebar of a freshly opened
 *  window did exactly nothing, visibly and indefinitely, and that is the bug that made
 *  the whole extension look broken.
 *
 *  A failure the caller can ignore by accident is a failure the user finds instead. */
export type Reply<T> = Partial<T> & { error?: string };

export class SidecarClient extends EventEmitter {
  private proc: ChildProcess | undefined;
  private accumulator = "";
  private nextId = 1;
  private pending = new Map<string, (result: unknown) => void>();
  // Last stderr lines, kept so a crash can be SURFACED rather than guessed at (S12.3).
  private stderrTail: string[] = [];

  get running(): boolean {
    return this.proc !== undefined && this.proc.exitCode === null;
  }

  start(binaryPath: string): void {
    if (this.running) {
      // Single instance (S12.3). Two sidecars means two models in unified memory,
      // which takes the machine down rather than degrading.
      throw new Error("sidecar already running");
    }
    this.accumulator = "";
    this.stderrTail = [];
    const proc = spawn(binaryPath, [], { stdio: ["pipe", "pipe", "pipe"] });
    this.proc = proc;

    proc.stdout?.on("data", (chunk: Buffer) => this.onStdout(chunk));
    proc.stderr?.on("data", (chunk: Buffer) => {
      for (const line of chunk.toString("utf8").split("\n")) {
        if (line.length === 0) continue;
        this.stderrTail.push(line);
        if (this.stderrTail.length > 40) this.stderrTail.shift();
      }
    });
    proc.on("exit", (code) => {
      this.proc = undefined;
      // Settle everything still outstanding BEFORE announcing the exit. A pending
      // request whose sidecar has died can never be answered, and an `await` on one
      // would hang the caller for the lifetime of the window -- including sidebar's
      // applyEdit, whose whole contract is that every path replies.
      const why = `sidecar exited (code ${code ?? "signal"})`;
      for (const [, resolve] of this.pending) resolve({ error: why });
      this.pending.clear();
      this.emit("exit", { code, stderr: this.stderrTail.join("\n") });
    });
  }

  /** Closes stdin. The sidecar exits on EOF, so no process is ever orphaned holding a
   *  model in memory (S12.3). */
  stop(): void {
    this.proc?.stdin?.end();
  }

  kill(): void {
    this.proc?.kill("SIGKILL");
  }

  private onStdout(chunk: Buffer): void {
    this.accumulator += chunk.toString("utf8");
    // Split on newline; anything after the last newline is an incomplete message and
    // stays buffered. Nothing inspects a partial message.
    let nl = this.accumulator.indexOf("\n");
    while (nl >= 0) {
      const message = this.accumulator.slice(0, nl);
      this.accumulator = this.accumulator.slice(nl + 1);
      if (message.length > 0) this.dispatch(message);
      nl = this.accumulator.indexOf("\n");
    }
  }

  private dispatch(raw: string): void {
    let msg: { id?: string; method?: string; params?: unknown; result?: unknown;
               error?: { message: string } };
    try {
      msg = JSON.parse(raw);
    } catch {
      // A malformed line is a transport defect, not a model failure. Say so.
      this.emit("exit", { code: null, stderr: `unparseable sidecar output: ${raw}` });
      return;
    }
    if (msg.id !== undefined && (msg.result !== undefined || msg.error !== undefined)) {
      const resolve = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      resolve?.(msg.error ? { error: msg.error.message } : msg.result);
      return;
    }
    if (msg.method === undefined) return;
    const event = msg.method.startsWith("lmp/") ? msg.method.slice(4) : msg.method;
    this.emit(event, msg.params);
  }

  private request<T extends object>(method: string, params: unknown): Promise<Reply<T>> {
    if (!this.proc?.stdin) {
      return Promise.resolve({
        error: `cannot send ${method}: the sidecar is not running`,
      } as Reply<T>);
    }
    const id = String(this.nextId++);
    const line = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    return new Promise((resolve) => {
      this.pending.set(id, resolve as (result: unknown) => void);
      this.proc?.stdin?.write(line);
    });
  }

  start_run(mission: string, settings: RunSettings): Promise<Reply<{ run_id: string }>> {
    const params: StartParams = { mission, settings };
    return this.request("lmp/start", params);
  }

  /** Loads the weights, as its own act.
   *
   *  Separate from start_run because it is the expensive, visible thing: ~19 GB and tens
   *  of seconds on a 48 GB machine. It used to be a side effect of the first mission, so
   *  the operator had no way to say WHEN it happened and no way to see that it was --
   *  the sidebar just read "Thinking" for a minute. Progress arrives as model_status
   *  notifications; this promise settles only once the load is over. */
  loadModel(modelDir: string): Promise<Reply<LoadModelResult>> {
    return this.request("lmp/load_model", { model_dir: modelDir });
  }

  /** Gives the memory back without killing the process. The conversation goes with it. */
  unloadModel(): Promise<Reply<{ unloaded: boolean }>> {
    return this.request("lmp/unload_model", {});
  }

  /** Deliverable while the model is generating: the sidecar's reader thread sets the
   *  cancel token as soon as the message is framed (S4.3). */
  cancel(runId: string): Promise<Reply<{ accepted: boolean }>> {
    return this.request("lmp/cancel", { run_id: runId });
  }

  approve(requestId: string, approved: boolean): Promise<Reply<{ accepted: boolean }>> {
    return this.request("lmp/approve", { request_id: requestId, approved });
  }

  /** The answer to an lmp/edit. The sidecar BLOCKS on this: `applied:false` carries the
   *  reason and becomes a tool error the model can act on, rather than a silent no-op. */
  editApplied(
    requestId: string,
    applied: boolean,
    error: string
  ): Promise<Reply<{ accepted: boolean }>> {
    return this.request("lmp/edit_applied", { request_id: requestId, applied, error });
  }

  /** Say something to the agent.
   *
   *  One method, two outcomes, and the caller does not choose between them: with a run
   *  in flight the text is steering and lands at the next turn boundary; with nothing
   *  running it is a follow-up and starts a run over the same conversation. `started_run`
   *  in the reply says which happened. */
  message(runId: string, text: string): Promise<Reply<MessageResult>> {
    return this.request("lmp/message", { run_id: runId, text });
  }

  shutdown(): Promise<Reply<{ ok: boolean }>> {
    return this.request("lmp/shutdown", {});
  }

  get protocolVersion(): string {
    return PROTOCOL_VERSION;
  }
}
