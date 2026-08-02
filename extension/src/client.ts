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
  RunEndNotification,
  TurnNotification,
  TokenNotification,
  ChecklistNotification,
  ApprovalRequestNotification,
  EditNotification,
  VerificationNotification,
  PerfNotification,
} from "./protocol.generated";

export interface SidecarEvents {
  token: TokenNotification;
  turn: TurnNotification;
  checklist: ChecklistNotification;
  verification: VerificationNotification;
  approval_request: ApprovalRequestNotification;
  edit: EditNotification;
  perf: PerfNotification;
  run_end: RunEndNotification;
  exit: { code: number | null; stderr: string };
}

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

  private request(method: string, params: unknown): Promise<unknown> {
    if (!this.proc?.stdin) return Promise.reject(new Error("sidecar is not running"));
    const id = String(this.nextId++);
    const line = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.proc?.stdin?.write(line);
    });
  }

  start_run(mission: string, settings: RunSettings): Promise<unknown> {
    const params: StartParams = { mission, settings };
    return this.request("lmp/start", params);
  }

  /** Deliverable while the model is generating: the sidecar's reader thread sets the
   *  cancel token as soon as the message is framed (S4.3). */
  cancel(runId: string): Promise<unknown> {
    return this.request("lmp/cancel", { run_id: runId });
  }

  approve(requestId: string, approved: boolean): Promise<unknown> {
    return this.request("lmp/approve", { request_id: requestId, approved });
  }

  /** The answer to an lmp/edit. The sidecar BLOCKS on this: `applied:false` carries the
   *  reason and becomes a tool error the model can act on, rather than a silent no-op. */
  editApplied(requestId: string, applied: boolean, error: string): Promise<unknown> {
    return this.request("lmp/edit_applied", { request_id: requestId, applied, error });
  }

  /** Say something to the agent.
   *
   *  One method, two outcomes, and the caller does not choose between them: with a run
   *  in flight the text is steering and lands at the next turn boundary; with nothing
   *  running it is a follow-up and starts a run over the same conversation. `started_run`
   *  in the reply says which happened. */
  message(runId: string, text: string): Promise<MessageResult> {
    return this.request("lmp/message", { run_id: runId, text }) as Promise<MessageResult>;
  }

  shutdown(): Promise<unknown> {
    return this.request("lmp/shutdown", {});
  }

  get protocolVersion(): string {
    return PROTOCOL_VERSION;
  }
}
