// The sidebar's markup, styles and view script (spec S12.2).
//
// Split out of sidebar.ts, which owns the protocol wiring: this file is the surface the
// user actually looks at, and mixing a stylesheet into the class that dispatches
// notifications made both harder to read.
//
// FOUR THINGS THIS FILE IS CAREFUL ABOUT
//
// 1. The sidecar streams. `Observer::on_token` now fires per token -- ~51 notifications a
//    turn -- so the typewriter is no longer reconstructing a stream from a slab, it is
//    smoothing arrival jitter over a real one. (It DID fire once per turn before `7cb3139`;
//    every "reconstructed at the view layer" assumption from that era is gone.) The drain
//    rate scales with how far behind the queue is, so a burst after a stall catches up.
//
// 2. Markdown is parsed incrementally, never re-parsed. A fenced block whose closing fence
//    has not arrived must not swallow the rest of the answer, and a chunk boundary must not
//    change the output -- which is what makes this a state machine and not a regex over the
//    accumulated text. The parser is `markdownStreamSource()`; the renderer is the only
//    part that touches the DOM. Keeping them apart is what lets the parser be diffed
//    against the C++ it was ported from -- see `scripts/verify-markdown-stream.js`.
//
// 3. Everything is themed from VS Code variables, so it follows the editor into light,
//    dark and high contrast rather than looking correct in whichever one it was built in.
//
// 4. The CSP allows inline styles and exactly one nonced script. No network, no CDN
//    fonts, no remote anything -- the sidecar is local and the view has no business
//    reaching off the machine. Nothing here ever sets innerHTML from model output: every
//    byte the model produced reaches the DOM as a text node.

import { orbStyles, orbMarkup, orbScript } from "./orb";

/** Font stack: the platform's own UI face first, so it reads native on macOS and does
 *  not fall back to something heavier elsewhere. */
const FONT_STACK =
  '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", ' +
  '"Segoe UI", Inter, system-ui, sans-serif';

function styles(): string {
  return `
:root {
  --r: 10px;
  --r-sm: 6px;
  --pad: 12px;
  --ease: cubic-bezier(.22,.68,.24,1);
  --accent: var(--vscode-textLink-foreground, #3b82f6);
  --fg: var(--vscode-foreground);
  --dim: color-mix(in srgb, var(--fg) 58%, transparent);
  --faint: color-mix(in srgb, var(--fg) 34%, transparent);
  --line: color-mix(in srgb, var(--fg) 12%, transparent);
  --surface: color-mix(in srgb, var(--fg) 4%, transparent);
  --surface-hi: color-mix(in srgb, var(--fg) 7%, transparent);
  --ok: #34a853;
  --fail: #ea4335;
  --warn: #f9ab00;
  --refuse: #a142f4;
}
* { box-sizing: border-box; }
body {
  font-family: ${FONT_STACK};
  font-size: 13px; line-height: 1.55;
  color: var(--fg); background: transparent;
  margin: 0; padding: 0;
  display: flex; flex-direction: column; height: 100vh;
  -webkit-font-smoothing: antialiased;
}

/* --- header ------------------------------------------------------------- */
#head {
  position: sticky; top: 0; z-index: 5;
  padding: 12px var(--pad) 10px;
  background: var(--vscode-sideBar-background);
  border-bottom: 1px solid var(--line);
}
#mission {
  font-size: 13px; font-weight: 590; letter-spacing: -.01em;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}
#mission:empty::before { content: "No run yet"; color: var(--faint); font-weight: 400; }
#status {
  display: flex; align-items: center; gap: 7px;
  margin-top: 6px; font-size: 11px; color: var(--dim);
  min-height: 16px;
}

/* --- the activity orb ---------------------------------------------------- */
/* A raymarched glass bead, in orb.ts. It replaced a 14px conic spinner: a spinner can
   only say "something is happening", and the orb says WHICH thing -- hue is the state,
   and the motion is sprung rather than looped, so a tool call reads differently from a
   pause. Unlike the spinner it is present at rest too, dimmed; an object that is always
   there and changes colour is not the same as an idle spinner nobody reads. */
${orbStyles()}

/* The label shimmers while busy -- the Gemini/Siri trick: a bright band swept across
   the text by animating a clipped gradient. */
body.busy #statusText {
  background: linear-gradient(90deg, var(--dim) 30%, var(--fg) 50%, var(--dim) 70%);
  background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: sweep 1.9s linear infinite;
}
@keyframes sweep { 0% { background-position: 120% 0; } 100% { background-position: -20% 0; } }

@media (prefers-reduced-motion: reduce) {
  body.busy #statusText { animation: none; -webkit-text-fill-color: var(--fg); }
}

/* --- checklist ----------------------------------------------------------- */
#plan { padding: 0 var(--pad); }
#plan:empty { display: none; }
#plan details {
  margin: 10px 0 0; background: var(--surface);
  border: 1px solid var(--line); border-radius: var(--r); overflow: hidden;
}
#plan summary {
  cursor: pointer; padding: 8px 10px; font-size: 11px; font-weight: 590;
  letter-spacing: .02em; text-transform: uppercase; color: var(--dim);
  list-style: none;
}
#plan summary::-webkit-details-marker { display: none; }
#plan ul { margin: 0; padding: 0 10px 10px; list-style: none; }
#plan li {
  display: flex; gap: 8px; align-items: flex-start; padding: 3px 0;
  animation: rise .3s var(--ease) both;
}
#plan li .box {
  flex: none; width: 14px; height: 14px; margin-top: 2px; border-radius: 4px;
  border: 1.5px solid var(--faint);
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; color: #fff; transition: all .25s var(--ease);
}
#plan li.done .box { background: var(--ok); border-color: var(--ok); }
#plan li.done .label { color: var(--dim); text-decoration: line-through; }

/* --- transcript ---------------------------------------------------------- */
#feed { flex: 1; overflow-y: auto; padding: 4px var(--pad) 8px; }
.msg { animation: rise .32s var(--ease) both; margin: 12px 0; }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

.msg.user {
  background: var(--surface-hi); border-radius: var(--r);
  padding: 9px 11px; white-space: pre-wrap; word-break: break-word;
  border-left: 2px solid var(--accent);
}
.msg.user .who {
  font-size: 10px; font-weight: 590; text-transform: uppercase;
  letter-spacing: .04em; color: var(--dim); margin-bottom: 3px;
}
.msg.assistant { white-space: pre-wrap; word-break: break-word; }

/* Rendered markdown. The state machine in markdownStreamSource() decides what the events
   are; these decide what they look like. Everything is themed from the same variables as
   the rest of the view, so code blocks follow the editor rather than shipping their own
   palette. */
.md-h { margin: 10px 0 2px; font-weight: 650; line-height: 1.3; }
h1.md-h { font-size: 1.3em; }
h2.md-h { font-size: 1.17em; }
h3.md-h { font-size: 1.06em; }
h4.md-h, h5.md-h, h6.md-h { font-size: 1em; color: var(--dim); }
.md-code {
  margin: 8px 0; padding: 8px 10px;
  background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--r-sm); overflow-x: auto;
  white-space: pre; word-break: normal;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
  line-height: 1.45;
}
.md-lang {
  display: block; font-family: ${FONT_STACK};
  font-size: 10px; font-weight: 590; text-transform: uppercase;
  letter-spacing: .04em; color: var(--faint); margin-bottom: 5px;
}
.md-inline {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em;
  background: var(--surface-hi); border-radius: 3px; padding: 1px 4px;
  word-break: break-all;
}
.md-list { margin: 3px 0; padding-left: 20px; }
.md-list li { margin: 1px 0; }
.md-list .md-list { margin: 1px 0; }
.caret {
  display: inline-block; width: 6px; height: 1.05em; margin-left: 1px;
  vertical-align: text-bottom; background: var(--accent); border-radius: 1px;
  animation: blink 1s steps(2) infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* Reasoning, behind a disclosure. Peeled off the answer by TOKEN ID upstream and never
   inlined into it (S5.7); this only decides how it is shown. */
.thought { margin: 8px 0; }
.thought summary {
  cursor: pointer; font-size: 11px; color: var(--dim); list-style: none;
  display: inline-flex; align-items: center; gap: 5px;
}
.thought summary::-webkit-details-marker { display: none; }
.thought summary::before { content: "✦"; color: var(--accent); font-size: 10px; }
.thought .body {
  margin-top: 6px; padding: 8px 10px; font-size: 12px; color: var(--dim);
  background: var(--surface); border-radius: var(--r-sm);
  white-space: pre-wrap; word-break: break-word;
}

/* --- tool rows ----------------------------------------------------------- */
.tool {
  border: 1px solid var(--line); border-radius: var(--r);
  background: var(--surface); overflow: hidden;
}
.tool summary {
  cursor: pointer; list-style: none; padding: 8px 10px;
  display: flex; align-items: center; gap: 8px; font-size: 12px;
}
.tool summary::-webkit-details-marker { display: none; }
.tool .name { font-weight: 590; }
.tool .args {
  color: var(--dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: var(--vscode-editor-font-family); font-size: 11px; flex: 1; min-width: 0;
}
.tool .ms { color: var(--faint); font-size: 10px; flex: none; }
.tool pre {
  margin: 0; padding: 0 10px 10px; font-size: 11px; white-space: pre-wrap;
  word-break: break-word; color: var(--dim);
  font-family: var(--vscode-editor-font-family); max-height: 260px; overflow: auto;
}
/* A policy refusal and a command failure are DIFFERENT FACTS. Conflating them is what
   sent v1's agent off "fixing" a build that was never run (S6.2, S12.2). */
.dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.dot.ok { background: var(--ok); }
.dot.failed { background: var(--fail); }
.dot.refused { background: var(--refuse); }

/* --- verification -------------------------------------------------------- */
.verify {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 8px 10px; border-radius: var(--r); background: var(--surface);
  border: 1px solid var(--line); font-size: 12px;
}
.pill {
  font-size: 10px; font-weight: 700; letter-spacing: .04em; padding: 2px 7px;
  border-radius: 999px; color: #fff; flex: none;
}
.pill.ok { background: var(--ok); }
.pill.fail { background: var(--fail); }
.verify code {
  font-family: var(--vscode-editor-font-family); font-size: 11px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0;
}
/* A green never shown capable of red is labelled UNPROVEN in the UI as well as in the
   ledger (S10.2) -- the user sees the same distinction the loop enforces. */
.unproven {
  color: var(--warn); font-size: 11px; width: 100%;
  border-top: 1px dashed color-mix(in srgb, var(--warn) 40%, transparent); padding-top: 5px;
}

/* --- approval cards ------------------------------------------------------ */
.card {
  border: 1px solid color-mix(in srgb, var(--warn) 45%, var(--line));
  background: color-mix(in srgb, var(--warn) 7%, transparent);
  border-radius: var(--r); padding: 11px; box-shadow: 0 2px 12px rgba(0,0,0,.14);
}
.card h4 { margin: 0 0 6px; font-size: 12px; display: flex; align-items: center; gap: 7px; }
.card .risk {
  margin-left: auto; font-size: 10px; font-weight: 700; padding: 2px 7px;
  border-radius: 999px; background: var(--warn); color: #1a1a1a;
}
.caps { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.cap {
  font-size: 10px; padding: 2px 6px; border-radius: var(--r-sm);
  background: color-mix(in srgb, var(--fail) 16%, transparent);
  border: 1px solid color-mix(in srgb, var(--fail) 34%, transparent);
}
.cap.parse { background: var(--surface-hi); border-color: var(--line); color: var(--dim); }
.card pre {
  margin: 6px 0; padding: 8px; background: var(--vscode-editor-background);
  border-radius: var(--r-sm); font-size: 11px; white-space: pre-wrap;
  word-break: break-word; font-family: var(--vscode-editor-font-family);
  max-height: 150px; overflow: auto;
}
.card .row { display: flex; gap: 8px; margin-top: 8px; }
button {
  font-family: inherit; font-size: 12px; font-weight: 590; cursor: pointer;
  border: none; border-radius: var(--r-sm); padding: 7px 14px;
  transition: transform .12s var(--ease), filter .12s var(--ease);
}
button:hover { filter: brightness(1.12); }
button:active { transform: scale(.97); }
button.primary { background: var(--accent); color: #fff; flex: 1; }
button.ghost {
  background: transparent; color: var(--fg);
  border: 1px solid var(--line); flex: 1;
}

.ended {
  text-align: center; font-size: 11px; color: var(--dim); padding: 10px 0 2px;
  border-top: 1px solid var(--line); margin-top: 14px;
}
.ended b { color: var(--fg); font-weight: 590; }

/* --- composer ------------------------------------------------------------ */
/* --- settings drawer ------------------------------------------------------ */
/* Live controls for the things worth changing between runs. They write straight back
   to the editor's own settings, so the drawer and the Settings UI are the same state
   rather than two copies that drift. */
#gear {
  position: absolute; top: 10px; right: 10px;
  width: 24px; height: 24px; padding: 0; border-radius: 6px;
  background: transparent; color: var(--dim); font-size: 13px;
}
#gear:hover { background: var(--surface-hi); color: var(--fg); }
#head { position: sticky; }
#drawer {
  display: none; padding: 10px 0 2px; border-top: 1px solid var(--line); margin-top: 10px;
}
#drawer.open { display: block; animation: rise .22s var(--ease) both; }
.set { margin-bottom: 11px; }
.set > label {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 11px; color: var(--dim); margin-bottom: 4px;
}
.set > label b { color: var(--fg); font-weight: 590; font-variant-numeric: tabular-nums; }
.seg { display: flex; gap: 3px; background: var(--surface); border-radius: 8px; padding: 2px; }
.seg button {
  flex: 1; padding: 4px 6px; font-size: 11px; border-radius: 6px;
  background: transparent; color: var(--dim); font-weight: 500;
}
.seg button.on { background: var(--vscode-sideBar-background); color: var(--fg); font-weight: 590;
                 box-shadow: 0 1px 3px rgba(0,0,0,.16); }
input[type=range] { width: 100%; accent-color: var(--accent); height: 16px; }
.toggle { display: flex; align-items: center; justify-content: space-between; padding: 5px 0; }
.toggle span { font-size: 12px; }
.toggle .sw {
  width: 34px; height: 19px; border-radius: 999px; background: var(--faint);
  position: relative; cursor: pointer; transition: background .2s var(--ease); flex: none;
}
.toggle .sw::after {
  content: ""; position: absolute; top: 2px; left: 2px; width: 15px; height: 15px;
  border-radius: 50%; background: #fff; transition: transform .2s var(--ease);
}
.toggle .sw.on { background: var(--accent); }
.toggle .sw.on::after { transform: translateX(15px); }
.warnbox {
  font-size: 10px; color: var(--warn); margin-top: 4px; line-height: 1.4;
}
#promptBox {
  width: 100%; min-height: 66px; max-height: 200px; resize: vertical;
  background: var(--vscode-input-background); color: var(--vscode-input-foreground);
  border: 1px solid var(--line); border-radius: var(--r-sm); padding: 6px;
  font-family: var(--vscode-editor-font-family); font-size: 11px; outline: none;
}
#promptBox:focus { border-color: var(--accent); }

#foot {
  padding: 8px var(--pad) 10px;
  background: var(--vscode-sideBar-background);
  border-top: 1px solid var(--line);
}
#perf {
  font-family: var(--vscode-editor-font-family); font-size: 10px;
  color: var(--faint); margin-bottom: 6px; min-height: 13px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
#composer {
  display: flex; align-items: flex-end; gap: 6px;
  background: var(--vscode-input-background);
  border: 1px solid var(--line); border-radius: 20px; padding: 4px 4px 4px 12px;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease);
}
#composer:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
}
#say {
  flex: 1; border: none; outline: none; resize: none; padding: 6px 0;
  background: transparent; color: var(--vscode-input-foreground);
  font-family: inherit; font-size: 13px; line-height: 1.45;
  max-height: 140px; overflow-y: auto;
}
#say::placeholder { color: var(--faint); }
#send {
  flex: none; width: 28px; height: 28px; padding: 0; border-radius: 50%;
  background: var(--accent); color: #fff; font-size: 14px; line-height: 1;
  display: flex; align-items: center; justify-content: center;
}
#send:disabled { opacity: .3; cursor: default; }
#hint { font-size: 10px; color: var(--faint); padding: 5px 4px 0; text-align: center; }
`;
}

function markup(): string {
  return `
<div id="head">
  <button id="gear" title="Settings">⚙</button>
  <div id="headRow">
    ${orbMarkup()}
    <div id="headText">
      <div id="mission"></div>
      <div id="status"><span id="statusText">Idle</span></div>
    </div>
  </div>
  <div id="drawer">
    <div class="set">
      <label>Mode</label>
      <div class="seg" id="segMode">
        <button data-v="plan">Plan</button>
        <button data-v="debug">Debug</button>
        <button data-v="agent">Agent</button>
      </div>
    </div>
    <div class="set">
      <label>Containment</label>
      <div class="seg" id="segTier">
        <button data-v="0">None</button>
        <button data-v="1">Sandbox</button>
        <button data-v="3">Host</button>
      </div>
      <div class="warnbox" id="tierWarn"></div>
    </div>
    <div class="toggle"><span>Run commands without asking</span><div class="sw" id="swExec"></div></div>
    <div class="toggle"><span>Write files without asking</span><div class="sw" id="swWrite"></div></div>
    <div class="set" id="sliders"></div>
    <div class="set">
      <label>System prompt <b id="promptMode"></b></label>
      <textarea id="promptBox" placeholder="Empty uses the built-in Piper persona"></textarea>
    </div>
  </div>
</div>
<div id="plan"></div>
<div id="feed"></div>
<div id="foot">
  <div id="perf"></div>
  <div id="composer">
    <textarea id="say" rows="1" placeholder="Message the agent…"></textarea>
    <button id="send" title="Send">↑</button>
  </div>
  <div id="hint"></div>
</div>`;
}

/** The incremental markdown state machine, as view-script source.
 *
 *  Ported from the winner of the MarkdownStream cook-off (Brief E, entrant e4 plus the
 *  nested-list fix) -- see `bakeoff/markdown_stream/README.md` for the scoreboard that
 *  chose it. Bytes in, render events out, no DOM: the renderer below is the only part
 *  that touches the document, so the parser stays checkable against the C++ original.
 *
 *  Exported so `scripts/verify-markdown-stream.js` can eval THIS text and diff it against
 *  the C++ amalgam on the cook-off corpus. Verifying a copy would verify the copy.
 *
 *  Two deliberate deviations from the C++, both because JS strings are UTF-16 and the
 *  original counted bytes:
 *    - `one()` never splits a surrogate pair when the holdback bound forces a flush.
 *    - a literal backtick never appears in this source; it is `TICK`, so the whole machine
 *      survives being carried inside a TypeScript template literal.
 */
export function markdownStreamSource(): string {
  return `
const TICK = String.fromCharCode(96);
const MAX_HOLDBACK = 64;

class MarkdownStream {
  constructor() { this.reset(); }

  reset() {
    this.hold = '';
    this.state = 'normal';
    this.atLineStart = true;
    this.listIndents = [];
    this.fencedTicks = 0;
    this.fencedInfo = '';
    this.inlineTicks = 0;
    this.headingLevel = 0;
    this.events = [];
  }

  pending() { return this.hold.length > 0; }

  // One unit of input to flush when the holdback bound is hit -- two if that would
  // otherwise tear a surrogate pair in half.
  one() {
    const c = this.hold.charCodeAt(0);
    return (c >= 0xD800 && c <= 0xDBFF && this.hold.length > 1) ? 2 : 1;
  }

  emitText(t) {
    if (!t) return;
    const last = this.events[this.events.length - 1];
    if (last && last.kind === 'text') last.text += t;
    else this.events.push({ kind: 'text', text: t });
  }

  emitCode(t) {
    if (!t) return;
    const last = this.events[this.events.length - 1];
    if (last && last.kind === 'codeText') last.text += t;
    else this.events.push({ kind: 'codeText', text: t });
  }

  closeLists(target) {
    while (this.listIndents.length && this.listIndents[this.listIndents.length - 1] >= target) {
      this.events.push({ kind: 'listClose', level: this.listIndents.length - 1 });
      this.listIndents.pop();
    }
  }

  closeAllLists() { this.closeLists(0); }

  feed(chunk) {
    this.hold += chunk;
    this.events = [];
    this.process(false);
    return this.events;
  }

  finish() {
    this.events = [];
    this.process(true);
    if (this.state === 'heading') this.events.push({ kind: 'headingClose', level: this.headingLevel });
    else if (this.state === 'fenceInfo' || this.state === 'fenceBody') this.events.push({ kind: 'codeClose' });
    else if (this.state === 'inline') this.events.push({ kind: 'inlineClose' });
    this.closeAllLists();
    const out = this.events;
    this.reset();
    return out;
  }

  process(isFinish) {
    while (this.hold.length) {
      if (this.state === 'normal') {
        if (this.atLineStart) {
          let spaces = 0;
          while (spaces < this.hold.length && this.hold[spaces] === ' ' && spaces < 32) spaces++;

          if (spaces === this.hold.length) {
            if (isFinish) { this.emitText(this.hold); this.hold = ''; }
            else if (this.hold.length >= MAX_HOLDBACK) {
              const n = this.one(); this.emitText(this.hold.slice(0, n)); this.hold = this.hold.slice(n);
            }
            return;
          }

          const c = this.hold[spaces];

          if (c === '\\n') {
            this.closeAllLists();
            this.events.push({ kind: 'paraBreak' });
            this.hold = this.hold.slice(spaces + 1);
            this.atLineStart = true;
            continue;
          }

          if (c === '#') {
            let hashes = 0, i = spaces;
            while (i < this.hold.length && this.hold[i] === '#' && hashes < 6) { hashes++; i++; }
            if (i === this.hold.length) {
              if (isFinish) { this.emitText(this.hold); this.hold = ''; }
              else if (this.hold.length >= MAX_HOLDBACK) {
                const n = this.one(); this.emitText(this.hold.slice(0, n));
                this.hold = this.hold.slice(n); this.atLineStart = false;
              }
              return;
            }
            if (this.hold[i] === ' ') {
              this.closeAllLists();
              this.headingLevel = hashes;
              this.events.push({ kind: 'headingOpen', level: hashes });
              this.state = 'heading';
              this.hold = this.hold.slice(i + 1);
              this.atLineStart = false;
              continue;
            } else if (this.hold[i] === '\\n') {
              this.closeAllLists();
              this.events.push({ kind: 'headingOpen', level: hashes });
              this.events.push({ kind: 'headingClose', level: hashes });
              this.hold = this.hold.slice(i + 1);
              this.atLineStart = true;
              continue;
            }
          }

          if (c === TICK) {
            let ticks = 0, i = spaces;
            while (i < this.hold.length && this.hold[i] === TICK) { ticks++; i++; }
            if (i === this.hold.length) {
              if (isFinish) { this.emitText(this.hold); this.hold = ''; }
              else if (this.hold.length >= MAX_HOLDBACK) {
                const n = this.one(); this.emitText(this.hold.slice(0, n));
                this.hold = this.hold.slice(n); this.atLineStart = false;
              }
              return;
            }
            if (ticks >= 3) {
              this.closeAllLists();
              this.fencedTicks = ticks;
              this.fencedInfo = '';
              // CodeBlockOpen is NOT pushed until the newline arrives: the info tag is not
              // known to be complete before it, and emitting early breaks split invariance.
              this.state = 'fenceInfo';
              this.hold = this.hold.slice(i);
              this.atLineStart = false;
              continue;
            }
          }

          if (c === '-' || c === '*') {
            if (spaces + 1 === this.hold.length) {
              if (isFinish) { this.emitText(this.hold); this.hold = ''; }
              return;
            }
            const nx = this.hold[spaces + 1];
            if (nx === ' ' || nx === '\\n') {
              this.closeLists(spaces + 2);
              this.events.push({ kind: 'listOpen', level: this.listIndents.length, ordered: false });
              this.listIndents.push(spaces + 2);
              this.hold = this.hold.slice(spaces + (nx === ' ' ? 2 : 1));
              this.atLineStart = (nx === '\\n');
              continue;
            }
          }

          if (c >= '0' && c <= '9') {
            let i = spaces;
            while (i < this.hold.length && this.hold[i] >= '0' && this.hold[i] <= '9') i++;
            if (i === this.hold.length) {
              if (i - spaces < 9 && !isFinish) {
                if (this.hold.length >= MAX_HOLDBACK) {
                  const n = this.one(); this.emitText(this.hold.slice(0, n));
                  this.hold = this.hold.slice(n); this.atLineStart = false;
                } else return;
              } else { this.emitText(this.hold); this.hold = ''; }
              return;
            } else if (this.hold[i] === '.') {
              if (i + 1 === this.hold.length) {
                if (isFinish) { this.emitText(this.hold); this.hold = ''; }
                return;
              }
              const nx = this.hold[i + 1];
              if (nx === ' ' || nx === '\\n') {
                this.closeLists(i + 2);
                this.events.push({
                  kind: 'listOpen', level: this.listIndents.length,
                  ordered: true, start: parseInt(this.hold.slice(spaces, i), 10),
                });
                this.listIndents.push(i + 2);
                this.hold = this.hold.slice(i + (nx === ' ' ? 2 : 1));
                this.atLineStart = (nx === '\\n');
                continue;
              }
            }
          }
        }

        let i = 0;
        while (i < this.hold.length && this.hold[i] !== TICK && this.hold[i] !== '\\n') i++;
        if (i > 0) {
          this.emitText(this.hold.slice(0, i));
          this.hold = this.hold.slice(i);
          this.atLineStart = false;
          continue;
        }
        if (!this.hold.length) return;

        if (this.hold[0] === '\\n') {
          this.emitText('\\n');
          this.hold = this.hold.slice(1);
          this.atLineStart = true;
          continue;
        }

        if (this.hold[0] === TICK) {
          let ticks = 0, j = 0;
          while (j < this.hold.length && this.hold[j] === TICK) { ticks++; j++; }
          if (j === this.hold.length) {
            if (isFinish) { this.emitText(this.hold); this.hold = ''; }
            else if (this.hold.length >= MAX_HOLDBACK) { this.emitText(TICK); this.hold = this.hold.slice(1); }
            return;
          }
          this.inlineTicks = ticks;
          this.events.push({ kind: 'inlineOpen' });
          this.state = 'inline';
          this.hold = this.hold.slice(j);
          this.atLineStart = false;
          continue;
        }
      } else if (this.state === 'heading') {
        const nl = this.hold.indexOf('\\n');
        if (nl < 0) {
          if (isFinish) { this.emitText(this.hold); this.hold = ''; }
          else if (this.hold.length >= MAX_HOLDBACK) {
            const n = this.one(); this.emitText(this.hold.slice(0, n)); this.hold = this.hold.slice(n);
          }
          return;
        }
        this.emitText(this.hold.slice(0, nl));
        this.events.push({ kind: 'headingClose', level: this.headingLevel });
        this.hold = this.hold.slice(nl + 1);
        this.state = 'normal';
        this.atLineStart = true;
        continue;
      } else if (this.state === 'fenceInfo') {
        const nl = this.hold.indexOf('\\n');
        if (nl < 0) {
          if (isFinish) {
            this.fencedInfo += this.hold;
            this.events.push({ kind: 'codeOpen', info: this.fencedInfo });
            this.hold = '';
            this.state = 'fenceBody';
            this.atLineStart = true;
          } else if (this.fencedInfo.length + this.hold.length >= MAX_HOLDBACK) {
            // An info tag that never ends is not an info tag. Give up on it as a language
            // hint, open the block, and let the bytes through as code rather than sitting
            // on the stream -- the C++ original drains one byte per feed here, which is a
            // stall the webview would show as a frozen bubble.
            this.events.push({ kind: 'codeOpen', info: this.fencedInfo });
            this.state = 'fenceBody';
            this.atLineStart = true;
            continue;
          }
          return;
        }
        this.fencedInfo += this.hold.slice(0, nl);
        this.events.push({ kind: 'codeOpen', info: this.fencedInfo });
        this.hold = this.hold.slice(nl + 1);
        this.state = 'fenceBody';
        this.atLineStart = true;
        continue;
      } else if (this.state === 'fenceBody') {
        if (this.atLineStart) {
          let spaces = 0;
          while (spaces < this.hold.length && this.hold[spaces] === ' ' && spaces < 32) spaces++;

          if (spaces === this.hold.length) {
            if (isFinish) { this.emitCode(this.hold); this.hold = ''; }
            else if (this.hold.length >= MAX_HOLDBACK) {
              const n = this.one(); this.emitCode(this.hold.slice(0, n));
              this.hold = this.hold.slice(n); this.atLineStart = false;
            }
            return;
          }

          if (this.hold[spaces] === TICK) {
            let ticks = 0, i = spaces;
            while (i < this.hold.length && this.hold[i] === TICK) { ticks++; i++; }
            if (i === this.hold.length) {
              if (isFinish) {
                if (ticks >= this.fencedTicks) {
                  this.events.push({ kind: 'codeClose' });
                  this.hold = ''; this.state = 'normal'; this.atLineStart = true;
                } else { this.emitCode(this.hold); this.hold = ''; }
              } else if (this.hold.length >= MAX_HOLDBACK) {
                const n = this.one(); this.emitCode(this.hold.slice(0, n));
                this.hold = this.hold.slice(n); this.atLineStart = false;
              }
              return;
            }
            if (ticks >= this.fencedTicks) {
              let j = i;
              while (j < this.hold.length && (this.hold[j] === ' ' || this.hold[j] === '\\t')) j++;
              if (j === this.hold.length) {
                if (isFinish) {
                  this.events.push({ kind: 'codeClose' });
                  this.hold = ''; this.state = 'normal'; this.atLineStart = true;
                } else if (this.hold.length >= MAX_HOLDBACK) {
                  const n = this.one(); this.emitCode(this.hold.slice(0, n));
                  this.hold = this.hold.slice(n); this.atLineStart = false;
                }
                return;
              }
              if (this.hold[j] === '\\n') {
                this.events.push({ kind: 'codeClose' });
                this.hold = this.hold.slice(j + 1);
                this.state = 'normal';
                this.atLineStart = true;
                continue;
              }
            }
          }

          if (this.hold[0] === '\\n') {
            this.emitCode('\\n');
            this.hold = this.hold.slice(1);
            this.atLineStart = true;
          } else {
            const n = this.one();
            this.emitCode(this.hold.slice(0, n));
            this.hold = this.hold.slice(n);
            this.atLineStart = false;
          }
          continue;
        } else {
          const nl = this.hold.indexOf('\\n');
          if (nl < 0) {
            if (isFinish) { this.emitCode(this.hold); this.hold = ''; }
            else if (this.hold.length >= MAX_HOLDBACK) {
              const n = this.one(); this.emitCode(this.hold.slice(0, n)); this.hold = this.hold.slice(n);
            }
            return;
          }
          this.emitCode(this.hold.slice(0, nl + 1));
          this.hold = this.hold.slice(nl + 1);
          this.atLineStart = true;
          continue;
        }
      } else if (this.state === 'inline') {
        let ticks = 0, j = 0;
        while (j < this.hold.length && this.hold[j] === TICK) { ticks++; j++; }
        if (j === this.hold.length) {
          if (isFinish) { this.emitText(this.hold); this.hold = ''; }
          else if (this.hold.length >= MAX_HOLDBACK) { this.emitText(TICK); this.hold = this.hold.slice(1); }
          return;
        }
        if (ticks === this.inlineTicks) {
          this.events.push({ kind: 'inlineClose' });
          this.state = 'normal';
          this.hold = this.hold.slice(ticks);
          continue;
        } else if (ticks > 0) {
          this.emitText(this.hold.slice(0, ticks));
          this.hold = this.hold.slice(ticks);
          continue;
        }
        let i = 0;
        while (i < this.hold.length && this.hold[i] !== TICK) i++;
        this.emitText(this.hold.slice(0, i));
        this.hold = this.hold.slice(i);
        continue;
      }
    }
  }
}
`;
}

/** The view script. Owns the typewriter queue, the busy state and the DOM. */
function script(): string {
  return `
${markdownStreamSource()}
${orbScript()}

const api = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);
const feed = $('feed');

let inFlight = false;
let bubble = null;          // the assistant bubble currently being typed into
let caret = null;
let mdCtx = null;           // the markdown parser + DOM cursor for that bubble

// --- typewriter -----------------------------------------------------------
// The sidecar now streams token by token, so this is no longer turning a slab into
// something readable -- it is smoothing arrival jitter. Keep it: the drain rate scales
// with how far behind it is, so a burst that lands after a stall still catches up rather
// than typing out at a leisurely pace, and a paused-then-resumed reader does not crawl.
const queue = [];
let typing = false;

function pump() { if (!typing) { typing = true; requestAnimationFrame(step); } }

function typeInto(node, text) { queue.push({ node, text, at: 0 }); pump(); }

// A structural job -- create a code block, open a list item. It runs from the SAME queue
// as the text rather than being applied on arrival, because the element a later Text event
// belongs in may not exist yet when that event is queued. Applying structure eagerly and
// text lazily puts the code block above the paragraph that introduced it.
function queueOp(fn) { queue.push({ op: fn }); pump(); }

// Markdown text resolves its destination at DRAIN time via ctx.target(), for the same
// reason: the container is whatever the structural jobs ahead of it have built.
function typeMd(ctx, text, trimLead) { queue.push({ ctx, text, at: 0, trimLead }); pump(); }

function step() {
  const job = queue[0];
  if (!job) { typing = false; return; }

  if (job.op) { job.op(); queue.shift(); requestAnimationFrame(step); return; }

  // One newline is dropped where prose resumes after a block element, so a code block or
  // heading does not leave a blank line behind it in the pre-wrap flow.
  if (job.ctx && job.at === 0 && job.trimLead && job.ctx.afterBlock) {
    job.ctx.afterBlock = false;
    if (job.text[0] === '\\n') job.text = job.text.slice(1);
    if (!job.text.length) { queue.shift(); requestAnimationFrame(step); return; }
  }

  const target = job.ctx ? job.ctx.target() : job.node;
  const left = job.text.length - job.at;
  const rate = Math.max(2, Math.ceil(left / 28));
  const chunk = job.text.slice(job.at, job.at + rate);
  job.at += rate;
  if (caret && caret.parentNode === target) {
    target.insertBefore(document.createTextNode(chunk), caret);
  } else {
    target.appendChild(document.createTextNode(chunk));
  }
  const pinned = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
  if (pinned) feed.scrollTop = feed.scrollHeight;
  if (job.at >= job.text.length) queue.shift();
  requestAnimationFrame(step);
}

// --- markdown rendering ---------------------------------------------------
// One MarkdownStream per assistant bubble. The parser says WHAT the events are; this says
// where in the DOM they land, and nothing else. Keeping the two apart is what lets the
// parser stay diffable against the C++ it was ported from.
function newMdCtx(bubbleEl) {
  const ctx = {
    bubble: bubbleEl,
    stream: new MarkdownStream(),
    lists: [],          // one entry per open nesting level: { ul, li }
    headEl: null, codeEl: null, inlineEl: null,
    afterBlock: false,
  };
  ctx.innerLi = () => (ctx.lists.length ? ctx.lists[ctx.lists.length - 1].li : null);
  ctx.blockParent = () => ctx.innerLi() || ctx.bubble;
  ctx.target = () => ctx.inlineEl || ctx.codeEl || ctx.headEl || ctx.blockParent();
  return ctx;
}

function applyMd(ctx, e) {
  if (e.kind === 'headingOpen') {
    ctx.lists.length = 0;
    const h = document.createElement('h' + Math.min(6, Math.max(1, e.level)));
    h.className = 'md-h';
    ctx.bubble.appendChild(h);
    ctx.headEl = h;
  } else if (e.kind === 'headingClose') {
    ctx.headEl = null; ctx.afterBlock = true;
  } else if (e.kind === 'codeOpen') {
    ctx.lists.length = 0;
    const pre = document.createElement('pre');
    pre.className = 'md-code';
    const lang = (e.info || '').trim().split(/\\s+/)[0];
    if (lang) {
      const tag = document.createElement('span');
      tag.className = 'md-lang';
      tag.textContent = lang;
      pre.appendChild(tag);
    }
    const code = document.createElement('code');
    pre.appendChild(code);
    ctx.bubble.appendChild(pre);
    ctx.codeEl = code;
  } else if (e.kind === 'codeClose') {
    ctx.codeEl = null; ctx.afterBlock = true;
  } else if (e.kind === 'inlineOpen') {
    const c = document.createElement('code');
    c.className = 'md-inline';
    ctx.target().appendChild(c);
    ctx.inlineEl = c;
  } else if (e.kind === 'inlineClose') {
    ctx.inlineEl = null;
  } else if (e.kind === 'listOpen') {
    // Depth is driven entirely by listOpen. listClose is ignored: the parser emits one
    // before every sibling item, so honouring it would start a fresh <ul> per bullet.
    if (e.level < ctx.lists.length) {
      const cur = ctx.lists[e.level];
      const li = document.createElement('li');
      cur.ul.appendChild(li);
      cur.li = li;
      ctx.lists.length = e.level + 1;
    } else {
      const ul = document.createElement(e.ordered ? 'ol' : 'ul');
      ul.className = 'md-list';
      if (e.ordered && e.start > 1) ul.setAttribute('start', String(e.start));
      ctx.blockParent().appendChild(ul);
      const li = document.createElement('li');
      ul.appendChild(li);
      ctx.lists.push({ ul, li });
    }
  } else if (e.kind === 'paraBreak') {
    ctx.lists.length = 0;
    if (ctx.afterBlock) ctx.afterBlock = false;
    else ctx.bubble.appendChild(document.createTextNode('\\n'));
  }
  if (caret) ctx.target().appendChild(caret);
}

function renderMd(ctx, events) {
  for (const e of events) {
    if (e.kind === 'text' || e.kind === 'codeText') typeMd(ctx, e.text, e.kind === 'text');
    else queueOp(((ev) => () => applyMd(ctx, ev))(e));
  }
}

// --- state ----------------------------------------------------------------
// The status line and the orb are set from the SAME call, so the word and the colour can
// never disagree about what the run is doing. Everything the orb knows arrives here.
function busy(on, label, state) {
  inFlight = on;
  document.body.classList.toggle('busy', on);
  $('statusText').textContent = label;
  if (state && window.__orb) window.__orb.state(state);
  $('hint').textContent = on
    ? 'Your message reaches the run at the next turn boundary'
    : 'Your message continues the conversation';
}

// sidebar.ts posts 'idle' in the same tick as 'run_end'. Left alone, that overwrites the
// terminal colour in the frame it started, and a run would end on the same grey it began
// on -- so the ending is held on screen before the orb is allowed to go back to rest.
let terminalUntil = 0;
let idleTimer = 0;

function finish(label, state) {
  terminalUntil = Date.now() + 2400;
  busy(false, label, state);
}

function goIdle() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => busy(false, 'Idle', 'IDLE'),
                         Math.max(0, terminalUntil - Date.now()));
}

function add(el, cls) {
  el.className = 'msg ' + (cls || '');
  feed.append(el);
  feed.scrollTop = feed.scrollHeight;
  return el;
}

// Ends BOTH streaming regions. Every section boundary (run_start, said, turn,
// verification, approval, run_end) already called this, so routing the thought through it
// means a new section can never land inside a still-open disclosure -- rather than six
// call sites each having to remember a second call.
function closeBubble() {
  // finish() flushes whatever the parser was still withholding and closes anything left
  // open, so an unterminated fence cannot swallow the answer. It is QUEUED, not applied
  // here, or it would land ahead of text still being typed.
  if (mdCtx) { renderMd(mdCtx, mdCtx.stream.finish()); mdCtx = null; }
  if (caret) {
    const c = caret;
    queueOp(() => { c.remove(); if (caret === c) caret = null; });
  }
  bubble = null;
  closeThought();
}

function openBubble() {
  closeBubble();
  bubble = add(document.createElement('div'), 'assistant');
  caret = document.createElement('span');
  caret.className = 'caret';
  bubble.append(caret);
  mdCtx = newMdCtx(bubble);
  return bubble;
}

// The one the streaming path wants. Tokens now arrive one at a time, so calling
// openBubble() per token would start a new assistant bubble for every token -- which is
// exactly what happened before the sidecar streamed, and went unnoticed because the
// notification only ever fired once per turn.
function currentBubble() {
  return bubble ?? openBubble();
}

// Reasoning gets ONE disclosure per turn, appended to, for the same reason.
let thought = null;         // the <details> body currently being streamed into
let thoughtSummary = null;

function closeThought() {
  if (thoughtSummary) thoughtSummary.textContent = 'Thought for a moment';
  thought = null;
  thoughtSummary = null;
}

function openThought() {
  if (thought) return thought;
  closeBubble();  // before the fields below are set: closeBubble() clears them
  const d = document.createElement('details');
  d.className = 'thought';
  thoughtSummary = document.createElement('summary');
  thoughtSummary.textContent = 'Thinking…';
  const b = document.createElement('div');
  b.className = 'body';
  d.append(thoughtSummary, b);
  add(d, '');
  thought = b;
  return b;
}

// --- composer -------------------------------------------------------------
const box = $('say');
const submit = () => {
  const text = box.value.trim();
  if (!text) return;
  api.postMessage({ kind: 'message', text });
  box.value = '';
  box.style.height = 'auto';
};
$('send').onclick = submit;
box.addEventListener('input', () => {
  box.style.height = 'auto';
  box.style.height = Math.min(box.scrollHeight, 140) + 'px';
});
box.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
});
busy(false, 'Idle', 'IDLE');

// --- settings drawer ------------------------------------------------------
// Every control writes back to the editor's own configuration, so this drawer and the
// Settings UI are one piece of state rather than two that drift.
const SLIDERS = [
  ['sampling.temperature', 'Temperature', 0, 2, 0.05,
   'Qwen3 thinking-mode default is 0.6. Lower repeats, higher loosens the tool grammar.'],
  ['sampling.topP', 'Top-p', 0.01, 1, 0.01, ''],
  ['sampling.topK', 'Top-k', 0, 100, 1, ''],
  ['sampling.minP', 'Min-p', 0, 0.5, 0.01, ''],
  ['sampling.repetitionPenalty', 'Repetition penalty', 1, 1.5, 0.01, ''],
];
let settings = {};

const put = (key, value) => {
  settings[key] = value;
  api.postMessage({ kind: 'setting', key, value });
};

$('gear').onclick = () => $('drawer').classList.toggle('open');

function seg(id, key, cast) {
  $(id).querySelectorAll('button').forEach((b) => {
    b.onclick = () => {
      put(key, cast(b.dataset.v));
      paint();
    };
  });
}
seg('segMode', 'mode', String);
seg('segTier', 'sandboxTier', Number);

function sw(id, key) {
  $(id).onclick = () => { put(key, !settings[key]); paint(); };
}
sw('swExec', 'autoApproveExec');
sw('swWrite', 'autoApproveWrites');

$('promptBox').addEventListener('change', () => {
  put('prompts.' + (settings.mode || 'agent'), $('promptBox').value);
});

function buildSliders() {
  const host = $('sliders');
  host.innerHTML = '';
  for (const [key, label, lo, hi, step] of SLIDERS) {
    const wrap = document.createElement('div');
    wrap.className = 'set';
    const l = document.createElement('label');
    const name = document.createElement('span');
    name.textContent = label;
    const val = document.createElement('b');
    l.append(name, val);
    const r = document.createElement('input');
    r.type = 'range'; r.min = lo; r.max = hi; r.step = step;
    r.oninput = () => { val.textContent = r.value; };
    r.onchange = () => put(key, Number(r.value));
    wrap.append(l, r);
    host.append(wrap);
    wrap.dataset.key = key;
  }
}
buildSliders();

function paint() {
  $('segMode').querySelectorAll('button').forEach(
    (b) => b.classList.toggle('on', b.dataset.v === settings.mode));
  $('segTier').querySelectorAll('button').forEach(
    (b) => b.classList.toggle('on', Number(b.dataset.v) === settings.sandboxTier));
  $('swExec').classList.toggle('on', settings.autoApproveExec === true);
  $('swWrite').classList.toggle('on', settings.autoApproveWrites === true);
  // Tier 2 is a real setting the segmented control has no button for; say so rather
  // than showing three buttons all unselected and looking broken.
  $('tierWarn').textContent =
    settings.sandboxTier === 3
      ? 'UNSANDBOXED: commands run with your permissions. No filesystem jail, no egress denial. You will be asked to confirm before the run starts.'
      : settings.sandboxTier === 0
      ? 'No execution at all. The agent can read and plan, not run.'
      : settings.sandboxTier === 2
      ? 'Container tier is selected in settings.json. It refuses until the runtime is wired.'
      : '';
  for (const [key] of SLIDERS) {
    const wrap = $('sliders').querySelector('[data-key="' + key + '"]');
    if (!wrap) continue;
    const v = settings[key];
    if (v === undefined) continue;
    wrap.querySelector('input').value = v;
    wrap.querySelector('b').textContent = v;
  }
  $('promptMode').textContent = '· ' + (settings.mode || 'agent');
  $('promptBox').value = settings['prompts.' + (settings.mode || 'agent')] || '';
}

// --- inbound --------------------------------------------------------------
window.addEventListener('message', (e) => {
  const { kind, payload } = e.data;

  if (kind === 'settings') { settings = payload; paint(); }

  if (kind === 'run_start') {
    $('mission').textContent = payload.mission;
    feed.textContent = ''; $('plan').textContent = ''; $('perf').textContent = '';
    closeBubble();
    busy(true, 'Thinking', 'THINKING');
  }

  if (kind === 'said') {
    closeBubble();
    const d = document.createElement('div');
    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = payload.steering ? 'You · mid-run' : 'You';
    const body = document.createElement('div');
    body.textContent = payload.text;
    d.append(who, body);
    add(d, 'user');
    busy(true, payload.steering ? 'Steering at the next turn' : 'Thinking', 'THINKING');
  }

  if (kind === 'token') {
    if (payload.channel === 'thinking') {
      typeInto(openThought(), payload.text);
      busy(true, 'Thinking', 'THINKING');
      if (window.__orb) window.__orb.impulse('token');
    } else {
      // Reasoning always precedes the answer within a turn, so the first answer token is
      // the signal that the thought is finished.
      closeThought();
      currentBubble();
      renderMd(mdCtx, mdCtx.stream.feed(payload.text));
      busy(true, 'Writing', 'WRITING');
      if (window.__orb) window.__orb.impulse('token');
    }
  }

  if (kind === 'turn') {
    closeBubble();
    const d = document.createElement('details');
    d.className = 'tool';
    const s = document.createElement('summary');
    const dot = document.createElement('span');
    dot.className = 'dot ' + (payload.tool_status === 'ok' ? 'ok'
      : (payload.tool_status === 'refused' || payload.tool_status === 'denied') ? 'refused' : 'failed');
    const n = document.createElement('span'); n.className = 'name'; n.textContent = payload.tool_name || payload.outcome;
    const a = document.createElement('span'); a.className = 'args'; a.textContent = payload.tool_args || '';
    const ms = document.createElement('span'); ms.className = 'ms';
    ms.textContent = payload.duration_ms ? Math.round(payload.duration_ms) + 'ms' : '';
    s.append(dot, n, a, ms);
    const pre = document.createElement('pre'); pre.textContent = payload.summary || '';
    d.append(s, pre);
    add(d, '');
    // The tool ROW is history; the orb is the live view of it. A call that failed or was
    // refused is not the same event as one that worked, and the colour says which.
    busy(true, 'Thinking',
         payload.tool_status === 'ok' || !payload.tool_status ? 'TOOL' : 'FAILED');
    if (window.__orb) window.__orb.impulse('tool');
  }

  if (kind === 'checklist') {
    const items = JSON.parse(payload.items_json);
    const done = items.filter((i) => i.done).length;
    $('plan').innerHTML = '';
    const det = document.createElement('details');
    det.open = true;
    const sum = document.createElement('summary');
    sum.textContent = 'Checklist · ' + done + '/' + items.length;
    const ul = document.createElement('ul');
    items.forEach((item, i) => {
      const li = document.createElement('li');
      li.className = item.done ? 'done' : '';
      li.style.animationDelay = (i * 28) + 'ms';
      const box = document.createElement('span');
      box.className = 'box';
      box.textContent = item.done ? '✓' : '';
      const label = document.createElement('span');
      label.className = 'label';
      label.textContent = item.text;
      li.append(box, label);
      ul.append(li);
    });
    det.append(sum, ul);
    $('plan').append(det);
  }

  if (kind === 'verification') {
    closeBubble();
    const d = document.createElement('div');
    d.className = 'verify';
    const p = document.createElement('span');
    p.className = 'pill ' + (payload.passed ? 'ok' : 'fail');
    p.textContent = payload.passed ? 'PASS' : 'FAIL';
    const c = document.createElement('code');
    c.textContent = payload.contract;
    d.append(p, c);
    if (payload.passed && !payload.falsifiable) {
      const u = document.createElement('div');
      u.className = 'unproven';
      u.textContent = 'UNPROVEN — this check has never been shown capable of failing, so it is not evidence yet.';
      d.append(u);
    }
    add(d, '');
    // A pass that has never been shown capable of failing is not evidence, and the orb does
    // not celebrate one: UNPROVEN gets the same amber the tool call got, not the green.
    busy(true, payload.passed ? 'Verified' : 'Verification failed',
         payload.passed ? (payload.falsifiable ? 'DONE' : 'TOOL') : 'FAILED');
  }

  if (kind === 'approval') {
    closeBubble();
    const card = document.createElement('div');
    card.className = 'card';
    const h = document.createElement('h4');
    h.textContent = 'Approve ' + payload.tool_name + '?';
    const risk = document.createElement('span');
    risk.className = 'risk';
    risk.textContent = 'risk ' + payload.risk.toFixed(2);
    h.append(risk);
    const caps = document.createElement('div');
    caps.className = 'caps';
    Object.entries(payload.capabilities).forEach(([k, v]) => {
      if (v !== true) return;
      const c = document.createElement('span'); c.className = 'cap'; c.textContent = k.replace(/_/g, ' ');
      caps.append(c);
    });
    const ps = payload.capabilities.parse_status;
    if (ps) {
      const c = document.createElement('span');
      c.className = 'cap parse';
      // "The full effect depends on bytes that are not in this string" is the single
      // most useful thing a card can say, and a risk scalar cannot say it.
      c.textContent = ps === 'parsed' ? 'fully parsed' : ps.replace(/_/g, ' ');
      caps.append(c);
    }
    const pre = document.createElement('pre');
    pre.textContent = payload.preview;
    const row = document.createElement('div');
    row.className = 'row';
    const yes = document.createElement('button'); yes.className = 'primary'; yes.textContent = 'Approve';
    const no = document.createElement('button'); no.className = 'ghost'; no.textContent = 'Deny';
    const answer = (ok, remember) => {
      api.postMessage({ kind: 'approve', id: payload.request_id, approved: ok, remember });
      card.remove();
    };
    yes.onclick = () => answer(true);
    no.onclick = () => answer(false);
    row.append(yes, no);

    // "Always allow" is offered only where it would actually do something. An
    // irreversible call escalates whatever is on the allowlist, so offering to remember
    // it would be a button that quietly does nothing -- worse than no button, because
    // the user would believe they had stopped being asked.
    if (payload.command && !payload.irreversible) {
      const always = document.createElement('button');
      always.className = 'ghost';
      always.textContent = 'Always allow';
      always.title = 'Remember this exact command: ' + payload.command;
      always.onclick = () => answer(true, payload.command);
      row.append(always);
    }
    card.append(h, caps, pre, row);
    if (payload.irreversible) {
      const note = document.createElement('div');
      note.className = 'warnbox';
      note.textContent =
        'This cannot be undone, so it always asks — no allowlist entry and no ' +
        '"run without asking" setting will skip this card.';
      card.append(note);
    }
    add(card, '');
    busy(true, 'Waiting for you', 'WAITING');
  }

  if (kind === 'perf') {
    const s = payload.sample;
    $('perf').textContent =
      'ttft ' + Math.round(s.ttft_ms) + 'ms · prefill ' + s.prefill_tok_per_s.toFixed(0) +
      ' tok/s · decode ' + s.decode_tok_per_s.toFixed(1) + ' tok/s · ctx ' +
      s.context_used + '/' + s.context_max;
  }

  if (kind === 'run_end') {
    closeBubble();
    const d = document.createElement('div');
    d.className = 'ended';
    let t = 'Ended: ' + payload.termination_reason + ' · ' + payload.iterations + ' turn(s)';
    // completed is EVIDENTIAL -- a recorded deliverable plus a falsifiable passing
    // verification. When it disagrees with the model's own checklist, both are shown
    // rather than whichever is more flattering (S10.4).
    if (payload.completed && payload.unfinished_items > 0) {
      t += ' · evidence says done, ' + payload.unfinished_items + ' item(s) left unticked';
    }
    // WHOSE criterion was met. "Complete" against a contract the model chose for itself
    // is a weaker claim than "Complete" against one you set, and both used to print the
    // same word -- a run once reported completed=yes on a mission it had not finished,
    // because the check it picked passed.
    if (payload.completed && payload.self_declared) {
      t += ' · against a check the model chose for itself';
    }
    const label = payload.completed
      ? (payload.self_declared ? 'Complete (self-checked)' : 'Complete')
      : 'Stopped';
    d.innerHTML = '<b>' + label + '</b> — ';
    d.append(document.createTextNode(t));
    feed.append(d);
    feed.scrollTop = feed.scrollHeight;
    // Green only for an ending that is actually complete. "Stopped" covers the wall clock,
    // the iteration cap and a cancel, and none of those are a success (S14).
    finish(label, payload.completed ? 'DONE' : 'FAILED');
  }

  if (kind === 'idle') goIdle();
});
`;
}

export function webviewHtml(nonce: string): string {
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>${styles()}</style></head><body>${markup()}
<script nonce="${nonce}">${script()}</script></body></html>`;
}
