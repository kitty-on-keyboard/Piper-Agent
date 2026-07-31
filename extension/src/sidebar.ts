// The sidebar webview (spec S12.2).
//
// Panels: chat with streaming answer; thinking behind a disclosure (reasoning is peeled
// off the answer body upstream, by TOKEN ID, and never inlined); a tool timeline where
// refused/denied is visually distinct from failed; the model's own checklist; the
// verification panel including falsifiability state; HITL approval cards with
// capability chips and a dry-run preview; and a perf HUD.

import * as vscode from "vscode";
import { SidecarClient } from "./client";
import {
  TokenNotification,
  TurnNotification,
  VerificationNotification,
  ApprovalRequestNotification,
  PerfNotification,
  RunEndNotification,
} from "./protocol.generated";

export class SidebarProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "lmPipe.sidebar";
  public currentRunId: string | undefined;
  private view: vscode.WebviewView | undefined;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly client: SidecarClient
  ) {
    this.client.on("token", (n: TokenNotification) => this.post("token", n));
    this.client.on("turn", (n: TurnNotification) => this.post("turn", n));
    this.client.on("verification", (n: VerificationNotification) => this.post("verification", n));
    this.client.on("perf", (n: PerfNotification) => this.post("perf", n));
    this.client.on("approval_request", (n: ApprovalRequestNotification) =>
      this.post("approval", n)
    );
    this.client.on("run_end", (n: RunEndNotification) => {
      // termination_reason is the one unambiguous signal for WHICH ENDING a run took
      // (S14). It is shown, not summarized away.
      this.post("run_end", n);
    });
  }

  beginRun(mission: string): void {
    this.post("run_start", { mission });
  }

  private post(kind: string, payload: unknown): void {
    void this.view?.webview.postMessage({ kind, payload });
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true, localResourceRoots: [this.extensionUri] };
    view.webview.html = this.html();
    view.webview.onDidReceiveMessage((msg: { kind: string; id?: string; approved?: boolean }) => {
      if (msg.kind === "approve" && msg.id !== undefined) {
        void this.client.approve(msg.id, msg.approved === true);
      }
      if (msg.kind === "cancel") {
        void vscode.commands.executeCommand("lmPipe.cancel");
      }
    });
  }

  private html(): string {
    const nonce = Math.random().toString(36).slice(2);
    return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
 body { font-family: var(--vscode-font-family); font-size: 12px; padding: 8px; }
 h3 { margin: 12px 0 4px; font-size: 11px; text-transform: uppercase; opacity: .7; }
 .row { display: flex; gap: 6px; align-items: baseline; padding: 3px 0;
        border-bottom: 1px solid var(--vscode-panel-border); }
 .chip { padding: 0 5px; border-radius: 3px; font-size: 10px; }
 /* Refused/denied is visually distinct from failed -- a policy refusal and a command
    failure are different facts, and conflating them is what sent v1's agent off
    "fixing" a build that was never run (S6.2, S12.2). */
 .ok { background: #1f6f3f; color: #fff; }
 .failed { background: #8b1a1a; color: #fff; }
 .refused { background: #6b4bab; color: #fff; }
 .unproven { color: #d9a441; }
 details { margin: 4px 0; opacity: .8; }
 #perf { font-family: var(--vscode-editor-font-family); opacity: .75; }
 .card { border: 1px solid var(--vscode-panel-border); padding: 6px; margin: 6px 0; }
 button { margin-right: 6px; }
</style></head><body>
<div id="mission"></div>
<h3>Answer</h3><div id="answer"></div>
<details id="thinkbox"><summary>Thinking</summary><div id="think"></div></details>
<h3>Tools</h3><div id="timeline"></div>
<h3>Checklist</h3><div id="checklist"></div>
<h3>Verification</h3><div id="verify"></div>
<h3>Approvals</h3><div id="approvals"></div>
<h3>Perf</h3><div id="perf"></div>
<script nonce="${nonce}">
const vscodeApi = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);
const chipFor = (status) =>
  status === 'ok' ? 'ok' : (status === 'refused' || status === 'denied') ? 'refused' : 'failed';

window.addEventListener('message', (e) => {
  const { kind, payload } = e.data;
  if (kind === 'run_start') {
    $('mission').textContent = payload.mission;
    ['answer','think','timeline','verify','approvals'].forEach(id => $(id).textContent = '');
  }
  if (kind === 'token') {
    const target = payload.channel === 'thinking' ? 'think' : 'answer';
    $(target).textContent += payload.text;
  }
  if (kind === 'turn') {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="chip ' + chipFor(payload.tool_status) + '">' +
      payload.tool_status + '</span><b>' + payload.tool_name + '</b>' +
      '<span style="opacity:.7">' + payload.tool_args + '</span>' +
      '<span style="margin-left:auto;opacity:.6">' + Math.round(payload.duration_ms) + 'ms</span>';
    const detail = document.createElement('details');
    detail.innerHTML = '<summary>result</summary><pre>' + payload.summary + '</pre>';
    $('timeline').append(row, detail);
  }
  if (kind === 'verification') {
    const d = document.createElement('div');
    d.className = 'row';
    // A green that has never been shown capable of red is labelled UNPROVEN, in the UI
    // as well as in the ledger (S10.2) -- the user sees the difference too.
    d.innerHTML = '<span class="chip ' + (payload.passed ? 'ok' : 'failed') + '">' +
      (payload.passed ? 'PASS' : 'FAIL') + '</span><code>' + payload.contract + '</code>' +
      (payload.passed && !payload.falsifiable
        ? '<span class="unproven">UNPROVEN: never shown capable of failing</span>' : '');
    $('verify').append(d);
  }
  if (kind === 'approval') {
    const caps = Object.entries(payload.capabilities)
      .filter(([k, v]) => v === true).map(([k]) => k).join(' ');
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '<b>' + payload.tool_name + '</b> risk ' + payload.risk.toFixed(2) +
      '<div>' + caps + '</div><pre>' + payload.preview + '</pre>';
    const yes = document.createElement('button'); yes.textContent = 'Approve';
    const no = document.createElement('button'); no.textContent = 'Deny';
    yes.onclick = () => { vscodeApi.postMessage({kind:'approve', id: payload.request_id, approved:true}); card.remove(); };
    no.onclick = () => { vscodeApi.postMessage({kind:'approve', id: payload.request_id, approved:false}); card.remove(); };
    card.append(yes, no);
    $('approvals').append(card);
  }
  if (kind === 'perf') {
    const s = payload.sample;
    $('perf').textContent = 'ttft ' + Math.round(s.ttft_ms) + 'ms  prefill ' +
      s.prefill_tok_per_s.toFixed(1) + ' tok/s  decode ' + s.decode_tok_per_s.toFixed(1) +
      ' tok/s  ctx ' + s.context_used + '/' + s.context_max;
  }
  if (kind === 'run_end') {
    const d = document.createElement('div');
    d.className = 'row';
    d.textContent = 'run ended: ' + payload.termination_reason +
      ' after ' + payload.iterations + ' iteration(s)';
    $('timeline').append(d);
  }
});
</script></body></html>`;
  }
}
