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
  '-apple-system, BlinkMacSystemFont, "Plus Jakarta Sans", Inter, "SF Pro Text", "Segoe UI Variable Text", ' +
  '"Segoe UI", system-ui, sans-serif';

function styles(): string {
  return `
:root {
  --r: 10px;
  --r-sm: 6px;
  --pad: 12px;
  --ease: cubic-bezier(.22,.68,.24,1);
  --accent: #14B8A6;
  --teal-blue-grad: linear-gradient(135deg, #14B8A6 0%, #06B6D4 40%, #3B82F6 75%, #8B5CF6 100%);
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

.wordmark {
  font-family: ${FONT_STACK};
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--fg);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
}
.wordmark #orb {
  --orb-size: 32px;
  margin-right: -7px;
}
.brand-iper {
  display: inline-flex;
  align-items: center;
}
.teal-i-wrapper {
  position: relative;
  display: inline-block;
}
.teal-dot-single {
  position: absolute;
  top: 3px;
  left: 50%;
  transform: translateX(-50%);
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #14B8A6;
  box-shadow: 0 0 8px #14B8A6;
}
.brand-agent {
  background: var(--teal-blue-grad);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
  letter-spacing: 0.01em;
  margin-left: 5px;
}

/* --- header ------------------------------------------------------------- */
#head {
  position: sticky; top: 0; z-index: 5;
  padding: 6px var(--pad) 8px;
  background: var(--vscode-sideBar-background);
  border-bottom: 1px solid var(--line);
}
#headRow {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 0 0 6px;
}
#headText {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
}
#mission {
  display: none !important;
}
#status {
  display: flex; align-items: center; justify-content: center; gap: 7px;
  margin-top: 4px; font-size: 11px; color: var(--dim);
  min-height: 16px;
}
/* The mode the RUN IN FLIGHT was started with. Not the segmented control's value, which
   is what the next run will use -- a mode changed mid-run would otherwise show the new
   word over the old behaviour, and the whole point of the word is to be trusted. */
#modeNow:not(:empty) {
  padding: 0 6px; border: 1px solid var(--line); border-radius: 9px;
  font-size: 10px; line-height: 15px; color: var(--faint); text-transform: uppercase;
  letter-spacing: .04em;
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
body.busy #statusText, body.busy #liveLabel {
  background: linear-gradient(90deg, var(--dim) 30%, var(--fg) 50%, var(--dim) 70%);
  background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: sweep 1.9s linear infinite;
}
@keyframes sweep { 0% { background-position: 120% 0; } 100% { background-position: -20% 0; } }

@media (prefers-reduced-motion: reduce) {
  body.busy #statusText, body.busy #liveLabel {
    animation: none; -webkit-text-fill-color: var(--fg);
  }
}

/* --- the model bar -------------------------------------------------------- */
/* WHY THIS IS ALWAYS VISIBLE. The sidecar holds ~19 GB of weights or it holds nothing,
   and until this existed there was no way to tell which from the outside -- so "no model
   is loaded" and "the model is thinking" looked identical: a status line, and no output.
   Loading is now an act with a button, a state and a duration, and all three are here. */
#modelBar {
  display: flex; align-items: center; gap: 7px;
  margin-top: 9px; padding-top: 9px; border-top: 1px solid var(--line);
  font-size: 11px; color: var(--dim); min-width: 0;
}
#modelName {
  flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: var(--vscode-editor-font-family); font-size: 11px;
  direction: rtl; text-align: left; /* a long path elides at the FRONT: the checkpoint
                                       name is the end of it, and the end is the part
                                       that tells you which model this is */
}
#modelBar button {
  flex: none; padding: 3px 9px; font-size: 11px; font-weight: 500;
  background: var(--surface-hi); color: var(--fg); border: 1px solid var(--line);
  border-radius: 999px;
}
#modelBar button:disabled { opacity: .45; cursor: default; }
#modelBar .dot.loading { background: var(--warn); animation: pulse 1.1s ease-in-out infinite; }
#modelBar .dot.unloaded { background: var(--faint); }
#modelSwitch { margin-top: 6px; }
#modelSwitch[hidden] { display: none !important; }
#modelSwitch button { font-size: 10px; padding: 4px 5px; min-width: 0; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
@keyframes pulse { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) { #modelBar .dot.loading { animation: none; } }

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
/* Rendered as ordinary chat, not as a monospace dump. Reasoning is prose with the same
   lists, code fences and emphasis the answer has, and showing it raw made the one part of
   the transcript people open out of curiosity the ugliest thing on screen. Same markdown
   pipeline, one notch dimmer so it still reads as an aside. */
.thought .body {
  margin-top: 6px; padding: 9px 11px; font-size: 12.5px; color: var(--dim);
  background: var(--surface); border-radius: var(--r-sm);
  border-left: 2px solid color-mix(in srgb, var(--accent) 35%, transparent);
  word-break: break-word;
}
.thought .body > :first-child { margin-top: 0; }
.thought .body > :last-child { margin-bottom: 0; }
/* The caret is the marker, and it turns. A disclosure that is shut by default has to look
   like one you can open. */
.thought summary::before {
  content: ""; width: 0; height: 0; flex: none;
  border-left: 4px solid var(--accent);
  border-top: 3.5px solid transparent; border-bottom: 3.5px solid transparent;
  transition: transform .18s var(--ease);
}
.thought[open] summary::before { transform: rotate(90deg); }
.thought summary:hover { color: var(--fg); }

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

/* --- question cards (Claude Code & Cursor style multi-select) --------------- */
.question-card {
  border: 1px solid color-mix(in srgb, var(--accent) 50%, var(--line));
  background: color-mix(in srgb, var(--accent) 8%, var(--surface));
  border-radius: var(--r);
  padding: 14px 16px;
  margin: 12px 0;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.28);
  display: flex;
  flex-direction: column;
  gap: 12px;
  backdrop-filter: blur(12px);
  transition: all 0.2s var(--ease);
}
.question-card.submitted {
  border-color: color-mix(in srgb, var(--ok) 50%, var(--line));
  background: color-mix(in srgb, var(--ok) 4%, var(--surface));
}
.q-header {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.question-card h4 {
  margin: 0;
  font-size: 13px;
  font-weight: 650;
  color: var(--fg);
  display: flex;
  align-items: center;
  gap: 8px;
  line-height: 1.4;
}
.question-card h4::before {
  content: '?';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--teal-blue-grad);
  color: #fff;
  font-size: 11px;
  font-weight: 800;
  flex: none;
  box-shadow: 0 0 10px color-mix(in srgb, var(--accent) 50%, transparent);
}
.q-subtitle {
  font-size: 11px;
  color: var(--dim);
  margin-left: 30px;
}
.q-options {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.q-opt-btn {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 13px;
  background: var(--surface-hi);
  border: 1px solid var(--line);
  border-radius: var(--r-sm);
  color: var(--fg);
  font-size: 12px;
  font-family: inherit;
  font-weight: 500;
  line-height: 1.45;
  cursor: pointer;
  text-align: left;
  transition: all 0.18s var(--ease);
  position: relative;
}
.q-opt-btn:hover {
  border-color: var(--accent);
  background: color-mix(in srgb, var(--accent) 14%, var(--surface-hi));
  transform: translateX(4px);
  box-shadow: 0 2px 10px color-mix(in srgb, var(--accent) 20%, transparent);
}
.q-opt-btn.selected {
  border-color: var(--accent);
  background: color-mix(in srgb, var(--accent) 22%, var(--surface-hi));
  box-shadow: 0 0 12px color-mix(in srgb, var(--accent) 30%, transparent);
}
.q-opt-badge {
  font-size: 11px;
  font-weight: 800;
  width: 20px;
  height: 20px;
  border-radius: 5px;
  background: color-mix(in srgb, var(--accent) 25%, transparent);
  color: var(--accent);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: none;
  transition: all 0.18s var(--ease);
  margin-top: 1px;
}
.q-opt-btn:hover .q-opt-badge, .q-opt-btn.selected .q-opt-badge {
  background: var(--teal-blue-grad);
  color: #fff;
  box-shadow: 0 0 8px color-mix(in srgb, var(--accent) 60%, transparent);
}
.q-opt-label {
  display: block;
}
.q-opt-detail {
  display: block;
  margin-top: 3px;
  font-size: 11px;
  font-weight: 400;
  line-height: 1.4;
  color: var(--dim);
}
.q-opt-text {
  flex: 1;
  word-break: break-word;
}
.q-footer {
  display: flex;
  justify-content: flex-end;
  margin-top: 4px;
}
.q-submit-btn {
  padding: 8px 16px;
  font-size: 12px;
  font-weight: 600;
  border-radius: var(--r-sm);
  background: var(--teal-blue-grad);
  color: #fff;
  border: none;
  cursor: pointer;
  box-shadow: 0 2px 12px color-mix(in srgb, var(--accent) 40%, transparent);
  transition: all 0.18s var(--ease);
}
.q-submit-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px color-mix(in srgb, var(--accent) 60%, transparent);
}


/* --- things that went wrong on THIS side ---------------------------------- */
/* A sidecar that will not start, a checkpoint that will not load, a request that could
   not be sent. All of these used to be discarded promises: the view simply stayed on
   "Thinking", which is indistinguishable from working and is the reason the extension
   looked dead rather than misconfigured. They belong in the feed, in sequence with
   everything else, because WHEN a failure happened is most of what it means. */
.msg.error {
  border: 1px solid color-mix(in srgb, var(--fail) 45%, var(--line));
  background: color-mix(in srgb, var(--fail) 8%, transparent);
  border-radius: var(--r); padding: 9px 11px; font-size: 12px;
  white-space: pre-wrap; word-break: break-word;
}
.msg.error .who {
  font-size: 10px; font-weight: 590; text-transform: uppercase;
  letter-spacing: .04em; color: var(--fail); margin-bottom: 3px;
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
#gear, #histBtn {
  position: absolute; top: 8px;
  width: 30px; height: 30px; padding: 0; border-radius: 8px;
  background: var(--surface); color: var(--fg); font-size: 16px;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.18s var(--ease);
  z-index: 10;
}
#gear { right: 10px; }
#histBtn { right: 50px; font-size: 17px; }
#gear:hover, #histBtn:hover { background: var(--surface-hi); color: var(--accent); transform: scale(1.08); }
#gear.on, #histBtn.on { background: var(--surface-hi); color: var(--accent); }
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
/* A control the loaded checkpoint cannot honour. Dimmed and inert, but still showing
   which value is selected -- the setting survives a model change, and hiding its value
   would leave the operator guessing what the next checkpoint will pick up. */
.seg.off { opacity: .5; }
.seg button:disabled { cursor: default; }
input[type=range] { width: 100%; accent-color: var(--accent); height: 16px; }
.toggle { display: flex; align-items: center; justify-content: space-between; padding: 5px 0; }
.toggle span { font-size: 12px; }
.toggle.off { opacity: .5; }
.toggle.off .sw { cursor: default; }
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
/* The same box saying a true thing that is not a problem: which of the two budgets is
   going to stop the run. Amber only when they disagree. */
.warnbox.calm { color: var(--faint); }
#promptBox {
  width: 100%; min-height: 66px; max-height: 200px; resize: vertical;
  background: var(--vscode-input-background); color: var(--vscode-input-foreground);
  border: 1px solid var(--line); border-radius: var(--r-sm); padding: 6px;
  font-family: var(--vscode-editor-font-family); font-size: 11px; outline: none;
}
#checkBox {
  width: 100%; background: var(--vscode-input-background); color: var(--vscode-input-foreground);
  border: 1px solid var(--line); border-radius: var(--r-sm); padding: 6px;
  font-family: var(--vscode-editor-font-family); font-size: 11px; outline: none;
}
#promptBox:focus { border-color: var(--accent); }

/* --- mode, on the main page ---------------------------------------------- */
/* Mode is not a preference, it is what the agent is ALLOWED to do this run: Plan cannot
   execute, Debug cannot write, Agent can do both. Burying that two clicks deep in a
   drawer put the single most consequential choice in the same place as the temperature
   slider. It lives on the surface now, and the drawer keeps the things you set once. */
#modeBar { margin-top: 9px; }
#modeBar .seg button { padding: 5px 6px; font-size: 11px; }

/* --- context meter -------------------------------------------------------- */
/* WHY THIS IS ON SCREEN. The context filling up is the single best predictor that answers
   are about to get worse, and compaction is the moment the run starts THROWING AWAY its
   own history. Both were invisible: "ctx 21378/100096" sat in a 10px monospace line among
   four other numbers, and compaction was not on the wire at all.
   The bar fills, warms through amber, and goes red near the top. */
#ctx {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 7px; font-size: 10px; color: var(--faint);
}
#ctxTrack {
  flex: 1; height: 3px; border-radius: 999px; min-width: 0;
  background: color-mix(in srgb, var(--fg) 10%, transparent);
  overflow: hidden;
}
#ctxFill {
  height: 100%; width: 0%; border-radius: 999px;
  background: var(--accent);
  transition: width .5s var(--ease), background-color .5s var(--ease);
}
#ctx.warm #ctxFill { background: var(--warn); }
#ctx.hot  #ctxFill { background: var(--fail); }
#ctxLabel { flex: none; font-variant-numeric: tabular-nums; }
#ctx.warm #ctxLabel { color: var(--warn); }
#ctx.hot  #ctxLabel { color: var(--fail); }
/* Compaction is an EVENT, not a level, so it gets a chip that appears when it happens
   and a pulse the first time each one lands. */
#ctxCompact {
  flex: none; display: none; align-items: center; gap: 4px;
  padding: 1px 7px; border-radius: 999px;
  background: color-mix(in srgb, var(--warn) 15%, transparent);
  border: 1px solid color-mix(in srgb, var(--warn) 38%, transparent);
  color: var(--warn); font-size: 10px; font-weight: 590;
}
#ctxCompact.on { display: inline-flex; }
#ctxCompact.bump { animation: chipPop .5s var(--ease); }
@keyframes chipPop { 0% { transform: scale(.8); opacity: 0; } 60% { transform: scale(1.06); } }
#ctx:empty, #ctx.idle { opacity: 0; }
#ctx { transition: opacity .3s var(--ease); }

/* --- run history ---------------------------------------------------------- */
#history { display: none; padding: 10px 0 2px; border-top: 1px solid var(--line); margin-top: 10px; }
#history.open { display: block; animation: rise .22s var(--ease) both; }
#history .empty { font-size: 11px; color: var(--faint); padding: 4px 0; }
/* Only rows that can actually be opened look like they can. */
#history .run.resumable { cursor: pointer; border-radius: 4px; }
#history .run.resumable:hover { background: rgba(127,127,127,.14); }
/* The third state: died mid-run. Not a failure and not a success -- the one to reopen. */
#history .rdot.dead { background: #d0a215; }
.run {
  display: flex; align-items: baseline; gap: 8px; padding: 7px 8px;
  border-radius: var(--r-sm); font-size: 11px;
}
.run + .run { margin-top: 2px; }
.run:hover { background: var(--surface-hi); }
.run .rdot { width: 6px; height: 6px; border-radius: 50%; flex: none; align-self: center; }
.run .rdot.ok { background: var(--ok); }
.run .rdot.no { background: var(--faint); }
.run .rmission {
  flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: var(--fg);
}
.run .rmeta { flex: none; color: var(--faint); font-variant-numeric: tabular-nums; }

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
/* --- attachments ---------------------------------------------------------
   A strip of chips above the composer, and a drop veil over the whole pane. The veil is
   pointer-events:none so it can never swallow a click when a stuck dragleave leaves it
   visible -- a drop target that eats the send button is worse than one that never shows. */
#attachments { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 10px; }
#attachments:empty { display: none; }
.attach-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 6px 3px 3px; border-radius: 6px;
  background: var(--vscode-badge-background); color: var(--vscode-badge-foreground);
  font-size: 11px; max-width: 220px;
}
.attach-chip img { width: 22px; height: 22px; object-fit: cover; border-radius: 4px; }
.attach-chip .n { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.attach-chip .x { cursor: pointer; opacity: 0.7; padding: 0 2px; }
.attach-chip .x:hover { opacity: 1; }
.attach-chip.pending { opacity: 0.55; }
.attach-chip.failed { background: var(--vscode-inputValidation-errorBackground); }
#dropVeil {
  position: fixed; inset: 0; display: none; pointer-events: none; z-index: 50;
  align-items: center; justify-content: center;
  background: color-mix(in srgb, var(--vscode-editor-background) 70%, transparent);
  border: 2px dashed var(--vscode-focusBorder);
  font-size: 13px; color: var(--vscode-foreground);
}
#dropVeil.on { display: flex; }
#composer {
  display: flex; align-items: flex-end; gap: 6px;
  background: var(--vscode-input-background);
  border: 1px solid var(--line); border-radius: 20px; padding: 4px 4px 4px 8px;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease);
}
/* --- the live row --------------------------------------------------------- */
/* The orb and the status word, INLINE AT THE TAIL OF THE TRANSCRIPT, which is where
   Claude and Gemini put theirs and where the eye already is: reading the last line of the
   answer. The orb used to sit in the composer and the word in the header, so the two
   halves of one signal were at opposite ends of the panel and the reader had to look away
   from the text to find out what was happening to it.

   A permanent last child of #feed rather than something appended and removed. It has to
   be permanent because the canvas holds a WebGL2 context: taking the row out of the DOM
   would lose it, and re-creating a context per run is both slow and a way to hit the
   browser's live-context cap. So the row stays and only its contents come and go. */
#live {
  display: flex; align-items: center; gap: 8px;
  padding: 2px 0 6px; min-height: 22px;
  font-size: 11px; color: var(--dim);
}
#live #orb { --orb-size: 18px; }
#live.idle #liveLabel { display: none; }
/* The toggle is the one part that stays visible at rest -- it is a preference, not a
   status, and hiding it between runs would mean it could only be found mid-run. */
#thinkToggle {
  margin-left: auto; flex: none;
  background: transparent; border: 1px solid var(--line); border-radius: 11px;
  padding: 1px 9px; font-size: 10px; line-height: 16px; color: var(--faint);
  cursor: pointer; transition: color .15s var(--ease), border-color .15s var(--ease);
}
#thinkToggle:hover { color: var(--dim); border-color: var(--dim); }
#thinkToggle.on { color: var(--accent); border-color: var(--accent); }
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
/* Stop REPLACES send while a run is turning, the way every other assistant does it.
   Two buttons side by side would mean the primary action changes meaning without moving,
   and the one you want mid-run is always the same one. */
#stop {
  flex: none; width: 28px; height: 28px; padding: 0; border-radius: 50%;
  background: var(--surface-hi); color: var(--fg); border: 1px solid var(--line);
  display: none; align-items: center; justify-content: center;
}
#stop::before {
  content: ""; width: 9px; height: 9px; border-radius: 2px; background: var(--fg);
}
#stop:hover { background: color-mix(in srgb, var(--fail) 18%, transparent);
              border-color: color-mix(in srgb, var(--fail) 45%, transparent); }
#stop:hover::before { background: var(--fail); }
body.busy #stop { display: flex; }
body.busy #send { display: none; }
#hint { font-size: 10px; color: var(--faint); padding: 5px 4px 0; text-align: center; }
`;
}

function markup(): string {
  return `
<div id="head">
  <button id="histBtn" title="Run history">◷</button>
  <button id="gear" title="Settings">⚙</button>
  <div id="headRow">
    <div id="headText">
      <div class="wordmark">
        ${orbMarkup()}
        <span class="brand-iper"><span class="teal-i-wrapper"><span class="teal-dot-single"></span>ı</span>per</span>
        <span class="brand-agent">Agent</span>
      </div>
      <div id="status"><span id="statusText">Idle</span><span id="modeNow"></span></div>
    </div>
  </div>
  <div id="modeBar">
    <div class="seg" id="segMode">
      <button data-v="agent">Agent</button>
      <button data-v="plan">Plan</button>
      <button data-v="debug">Debug</button>
    </div>
  </div>
  <div id="modelBar">
    <span id="modelDot" class="dot unloaded"></span>
    <span id="modelName">No model</span>
    <button id="modelAction">Load</button>
    <button id="modelPick" title="Choose a different model directory">Change</button>
  </div>
  <div id="modelSwitch" class="seg" hidden></div>
  <div id="history"></div>
  <div id="drawer">
    <div class="set">
      <label>Containment</label>
      <div class="seg" id="segTier">
        <button data-v="0">None</button>
        <button data-v="1">Sandbox</button>
        <button data-v="3">Host</button>
      </div>
      <div class="warnbox" id="tierWarn"></div>
    </div>
    <div class="set">
      <label>Thinking <b id="effortState"></b></label>
      <div class="seg" id="segEffort">
        <button data-v="low" title="Keep thinking brief and move to the conclusion">Low</button>
        <button data-v="medium" title="No instruction either way — the checkpoint's untouched behaviour">Medium</button>
        <button data-v="xhigh" title="Think carefully, validate assumptions, weigh alternatives — the slowest level">xHigh</button>
      </div>
      <div class="warnbox" id="effortWarn"></div>
    </div>
    <div class="toggle"><span>Run commands without asking</span><div class="sw" id="swExec"></div></div>
    <div class="toggle"><span>Write files without asking</span><div class="sw" id="swWrite"></div></div>
    <div class="toggle" id="specRow"><span>Speculative decoding</span><div class="sw" id="swSpec"></div></div>
    <div class="warnbox" id="specWarn"></div>
    <div class="set">
      <label>Check command <b id="checkState"></b></label>
      <input type="text" id="checkBox" placeholder="e.g. swift build — empty for no check">
      <div class="warnbox" id="checkWarn"></div>
    </div>
    <div class="set">
      <label>Turn limit <b id="turnVal"></b></label>
      <input type="range" id="turnRange" min="20" max="600" step="10">
      <label style="margin-top:8px">Time limit <b id="clockVal"></b></label>
      <input type="range" id="clockRange" min="300" max="43200" step="300">
      <label style="margin-top:8px">Stop if stuck for <b id="stallVal"></b></label>
      <input type="range" id="stallRange" min="60" max="7200" step="60">
      <div class="warnbox" id="budgetWarn"></div>
    </div>
    <div class="set" id="sliders"></div>
    <div class="set">
      <label>System prompt <b id="promptMode"></b></label>
      <textarea id="promptBox" placeholder="Empty uses the built-in Piper persona"></textarea>
    </div>
    <div class="set"><label>View build <b id="viewBuild"></b></label></div>
  </div>
</div>
<div id="plan"></div>
<div id="feed"><div id="live"><span id="liveOrb">${orbMarkup()}</span><span id="liveLabel"></span><button id="thinkToggle" title="Keep reasoning open in the chat">Thinking</button></div></div>
<div id="foot">
  <div id="ctx" class="idle">
    <div id="ctxTrack"><div id="ctxFill"></div></div>
    <span id="ctxLabel"></span>
    <span id="ctxCompact"></span>
  </div>
  <div id="perf"></div>
  <div id="dropVeil">Drop an image to show it to the agent</div>
  <div id="attachments"></div>
  <div id="composer">
    <textarea id="say" rows="1" placeholder="Message the agent…"></textarea>
    <button id="stop" title="Stop this run"></button>
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
// The permanent last child of the feed. Everything add() puts in the transcript goes
// BEFORE it, so the live status always trails the history it is reporting on.
const live = $('live');
// Whether reasoning disclosures open as they arrive. Mirrors lmPipe.showThinking, which
// is where it lives so that hiding the panel -- which destroys this webview -- does not
// forget it.
let showThinking = false;

let inFlight = false;
let bubble = null;          // the assistant bubble currently being typed into
let caret = null;
let mdCtx = null;           // the markdown parser + DOM cursor for that bubble
// Last turn's tool, so run_end can tell an \`ask_user\` halt from a Plan-mode pause.
// Both arrive as termination_reason \`awaiting_user\`; dressing the pause as a question
// is how "Your turn — answer in the box below" appeared after a list_dir.
let lastTurnTool = '';
let lastQuestionCardDrawn = false;

// --- typewriter -----------------------------------------------------------
//
// ONE QUEUE, AND EVERYTHING GOES THROUGH IT. Text, markdown structure, tool rows,
// verification rows, approval cards, the run footer -- all of it. That is what makes the
// transcript's order a property of arrival order instead of a race, and it is the fix for
// the reordering.
//
// The bug it replaces: blocks were appended to the feed SYNCHRONOUSLY, on arrival, while
// text drained from this queue at a fixed trickle. The comment here used to claim the
// drain rate "scales with how far behind it is". It did not. rate was computed from the
// length of the CURRENT JOB, and once the sidecar began streaming token by token every job
// held about one token -- four characters -- so Math.max(2, ceil(4/28)) pinned the rate
// at the floor of 2 characters per frame. Roughly 120 characters a second against a decode
// rate several times that. The queue fell behind at a rate proportional to how much the
// model had to say, and never caught up within a run, so a tool row raised at token 400
// appeared on screen while the reader was still watching sentence two arrive.
//
// The rate now comes from the WHOLE BACKLOG and is spent across as many jobs as it takes,
// so lag is bounded by a couple of frames no matter how long the answer is.
const queue = [];
let typing = false;
// Characters queued and not yet on screen, maintained incrementally -- the one number the
// old rate was missing. Walking the queue for it every frame would be O(queue) per frame
// on exactly the runs where the queue is longest.
let pending = 0;

// Above this, smoothing is a lie: the reader is watching a stale transcript, not a
// typewriter. A webview that was hidden, a burst after a stall, or a replayed slab all
// land here, and the honest response is to put it on screen at once.
const MAX_PENDING = 4000;

function pump() { if (!typing) { typing = true; requestAnimationFrame(step); } }

function typeInto(node, text) { queue.push({ node, text, at: 0 }); pending += text.length; pump(); }

// A structural job -- create a code block, open a list item, append a tool row. It runs
// from the SAME queue as the text rather than being applied on arrival, because the
// element a later Text event belongs in may not exist yet when that event is queued.
// Applying structure eagerly and text lazily puts the code block above the paragraph that
// introduced it -- and puts the whole tool row above the sentence that introduced IT.
function queueOp(fn) { queue.push({ op: fn }); pump(); }

// Markdown text resolves its destination at DRAIN time via ctx.target(), for the same
// reason: the container is whatever the structural jobs ahead of it have built.
function typeMd(ctx, text, trimLead) {
  queue.push({ ctx, text, at: 0, trimLead });
  pending += text.length;
  pump();
}

// Empties the queue to the screen in one go. For the boundaries where a partly-typed
// transcript would be wrong rather than merely slow: the user has just sent a message, or
// the run has ended and the footer is about to land.
function flushQueue() {
  let guard = 0;
  // Bounded rather than while (queue.length): a job whose op throws would otherwise spin
  // the tab forever, and a render bug should degrade to a jumpy transcript, not a hang.
  while (queue.length && ++guard < 10000) drain(Number.MAX_SAFE_INTEGER);
  // typing is deliberately NOT cleared here. A frame may already be scheduled, and
  // clearing it would let the next pump() schedule a second one -- two step() chains
  // draining one queue, each granting itself a full budget. The pending frame will find
  // the queue empty and stand itself down, which is the same thing one tick later.
  feed.scrollTop = feed.scrollHeight;
}

// WHERE THE NEXT THING GOES, for every writer into the transcript.
//
// The caret is the write cursor, so anything being added lands BEFORE it when it is a
// child of the container -- text and elements alike. There used to be two rules: drain()
// inserted text before the caret while applyMd() appended elements past it, and any
// container holding a caret that was not its last child split permanently in two, prose
// before and elements after.
//
// MEASURED: "identified it as a Swift package (\`DemoApp.xcodeproj\`, \`Package.swift\`)"
// rendered as "identified it as a Swift package (, )" with every code span swept to the
// end of the bubble and run together, since adjacent <code> siblings with no text between
// them read as a single span. applyMd re-anchors the caret after each structural op, which
// hid this whenever it ran -- but that re-anchoring is conditional on the context still
// being the current one, and a bubble draining after the next has opened is exactly the
// case where it stops. One rule cannot come apart that way.
//
// Pinned by scripts/verify-inline-code-order.js.
function putIn(parent, node) {
  if (caret && caret.parentNode === parent) parent.insertBefore(node, caret);
  else parent.appendChild(node);
  return node;
}

// Spends budget characters across as many jobs as it takes. Returns what it did not
// spend, so a frame that runs out of queue does not busy-wait on the remainder.
function drain(budget) {
  while (queue.length && budget > 0) {
    const job = queue[0];

    if (job.op) { job.op(); queue.shift(); continue; }

    // One newline is dropped where prose resumes after a block element, so a code block or
    // heading does not leave a blank line behind it in the pre-wrap flow.
    if (job.ctx && job.at === 0 && job.trimLead && job.ctx.afterBlock) {
      job.ctx.afterBlock = false;
      if (job.text[0] === '\\n') { job.text = job.text.slice(1); pending -= 1; }
      if (!job.text.length) { queue.shift(); continue; }
    }

    const target = job.ctx ? job.ctx.target() : job.node;
    const left = job.text.length - job.at;
    const take = Math.min(left, budget);
    const chunk = job.text.slice(job.at, job.at + take);
    job.at += take;
    budget -= take;
    pending -= take;
    putIn(target, document.createTextNode(chunk));
    if (job.at >= job.text.length) queue.shift();
  }
  return budget;
}

function step() {
  if (!queue.length) { typing = false; return; }

  // The floor keeps a short answer from arriving instantly -- the smoothing is still worth
  // having when there is nothing to catch up on. The divisor is what makes it catch up:
  // a backlog of 600 characters clears in about six frames rather than five seconds.
  const budget = pending > MAX_PENDING ? Number.MAX_SAFE_INTEGER
                                       : Math.max(24, Math.ceil(pending / 6));
  drain(budget);

  const pinned = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
  if (pinned) feed.scrollTop = feed.scrollHeight;
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
    putIn(ctx.bubble, h);
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
    putIn(ctx.bubble, pre);
    ctx.codeEl = code;
  } else if (e.kind === 'codeClose') {
    ctx.codeEl = null; ctx.afterBlock = true;
  } else if (e.kind === 'inlineOpen') {
    const c = document.createElement('code');
    c.className = 'md-inline';
    putIn(ctx.target(), c);
    ctx.inlineEl = c;
  } else if (e.kind === 'inlineClose') {
    ctx.inlineEl = null;
  } else if (e.kind === 'listOpen') {
    // Depth is driven entirely by listOpen. listClose is ignored: the parser emits one
    // before every sibling item, so honouring it would start a fresh <ul> per bullet.
    if (e.level < ctx.lists.length) {
      const cur = ctx.lists[e.level];
      const li = document.createElement('li');
      putIn(cur.ul, li);
      cur.li = li;
      ctx.lists.length = e.level + 1;
    } else {
      const ul = document.createElement(e.ordered ? 'ol' : 'ul');
      ul.className = 'md-list';
      if (e.ordered && e.start > 1) ul.setAttribute('start', String(e.start));
      putIn(ctx.blockParent(), ul);
      const li = document.createElement('li');
      ul.appendChild(li);
      ctx.lists.push({ ul, li });
    }
  } else if (e.kind === 'paraBreak') {
    ctx.lists.length = 0;
    if (ctx.afterBlock) ctx.afterBlock = false;
    else putIn(ctx.bubble, document.createTextNode('\\n'));
  }
  // ONLY for the context the caret actually belongs to.
  //
  // There is one caret and it is a module global, so this used to drag it into whatever
  // context happened to be draining. closeThought() queues the thought's tail-flush and
  // the answer path then synchronously opens a new bubble with a new caret -- and when
  // those queued thought ops drained, this line moved the ANSWER's caret into the thought
  // body, which is collapsed. Text still landed correctly (step() falls back to append
  // when the caret is elsewhere), so the only symptom was the blinking cursor vanishing
  // from the answer, which reads as the stream having stalled.
  if (caret && ctx === mdCtx) ctx.target().appendChild(caret);
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
  // The same word, in the transcript, next to the orb. Set from the SAME call as the hue
  // for the reason the header status was: two places that describe one thing drift, and
  // the one nobody is looking at is the one that goes stale.
  $('liveLabel').textContent = label;
  live.classList.toggle('idle', !on);
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

// ADDS classes; it used to assign them.
//
// Assigning el.className overwrote whatever the caller had already set, and four
// callers set one before calling: the reasoning disclosure (.thought), tool rows (.tool),
// verification rows (.verify) and approval cards (.card). Every one of them reached the
// feed as a bare .msg, so none of that CSS had ever applied to anything -- the thought
// disclosure had no marker and no body panel, and a tool row had no border. The styles
// were written, reviewed and shipped, and were dead on arrival at this line.
// QUEUED, not appended. This is the other half of the ordering fix.
//
// Every caller here -- the tool row, the verification row, the approval card, the error
// block, the user's echo, the reasoning disclosure -- used to land in the feed the instant
// its message arrived, while the answer that preceded it was still trickling out of the
// typewriter. The elements were in the right ORDER; the earlier one was still filling in
// after the later one was on screen, which is the same thing to read.
//
// Going through the queue means a block cannot overtake text that arrived before it, by
// construction rather than by the typewriter happening to keep up.
function add(el, cls) {
  el.classList.add('msg');
  if (cls) el.classList.add(cls);
  queueOp(() => {
    // Before the live row, never after it: #live is a permanent last child, and appending
    // past it would put the transcript underneath the thing that reports on it.
    feed.insertBefore(el, live);
    feed.scrollTop = feed.scrollHeight;
  });
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
let thoughtDetails = null;
let thoughtStarted = 0;
let thoughtMd = null;       // its markdown context -- same pipeline as the answer

// SHUT BY DEFAULT.
//
// It used to open itself while the tokens arrived, on the reasoning that watching them is
// the evidence the model is working. The orb and the shimmering status line do that job
// now, and they do it without moving the transcript: a disclosure that opens, fills with
// several hundred tokens and then collapses pushes the answer down the screen and yanks it
// back, on every single turn. Reasoning is an aside you open when you want it.
function closeThought() {
  // The parser holds bytes back mid-construct; finishing it is what flushes the tail and
  // closes an unterminated fence, exactly as the answer bubble does.
  if (thoughtMd) { renderMd(thoughtMd, thoughtMd.stream.finish()); thoughtMd = null; }
  if (thoughtSummary) {
    const secs = (Date.now() - thoughtStarted) / 1000;
    thoughtSummary.textContent =
      'Thought for ' + (secs < 1 ? 'a moment' : secs.toFixed(secs < 10 ? 1 : 0) + 's');
  }
  thought = null;
  thoughtSummary = null;
  thoughtDetails = null;
}

function openThought() {
  if (thought) return thought;
  closeBubble();  // before the fields below are set: closeBubble() clears them
  const d = document.createElement('details');
  d.className = 'thought';
  d.open = showThinking;
  thoughtSummary = document.createElement('summary');
  thoughtSummary.textContent = 'Thinking…';
  const b = document.createElement('div');
  b.className = 'body';
  d.append(thoughtSummary, b);
  add(d, '');
  thought = b;
  thoughtDetails = d;
  thoughtStarted = Date.now();
  // The SAME renderer the answer uses, so opening a thought shows prose, lists and code
  // fences rather than a wall of preformatted text. newMdCtx takes the element it writes
  // into, so this costs one line and no second pipeline to keep in step.
  thoughtMd = newMdCtx(b);
  return b;
}

// --- composer -------------------------------------------------------------
const box = $('say');
// --- attachments ----------------------------------------------------------
//
// Dropped and pasted images. The bytes go to the HOST, which writes them into the
// workspace and sends back a path -- the webview has no filesystem, and the sidecar only
// accepts paths inside the workspace jail anyway.
//
// A chip appears immediately, before the write finishes, and carries its own thumbnail
// from the local blob. Waiting for the round-trip to acknowledge a drop reads as the drop
// not having worked, and the user drops again.
let attachSeq = 0;
const attached = new Map();

function renderAttachments() {
  const strip = $('attachments');
  strip.innerHTML = '';
  for (const [id, a] of attached) {
    const chip = document.createElement('span');
    chip.className = 'attach-chip' + (a.path ? '' : a.error ? ' failed' : ' pending');
    if (a.thumb) {
      const img = document.createElement('img');
      img.src = a.thumb;
      chip.appendChild(img);
    }
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = a.error ? (a.name + ' — ' + a.error) : a.name;
    chip.appendChild(n);
    const x = document.createElement('span');
    x.className = 'x';
    x.textContent = '×';
    x.title = 'Remove';
    x.onclick = () => { attached.delete(id); renderAttachments(); };
    chip.appendChild(x);
    strip.appendChild(chip);
  }
}

function offerFile(file) {
  // indexOf, not a regex. THIS SCRIPT IS THE BODY OF A TEMPLATE LITERAL: a backslash in
  // it is consumed when the literal is evaluated, so /^image\\// arrived in the webview as
  // /^image//, which is a syntax error -- and a syntax error here does not break one
  // feature, it stops the whole script before any listener is attached, so every button
  // in the pane goes dead. Avoiding the escape entirely is worth more than the regex.
  if (!file || file.type.indexOf('image/') !== 0) return;
  const id = 'a' + (++attachSeq);
  attached.set(id, { name: file.name || 'pasted image', thumb: URL.createObjectURL(file) });
  renderAttachments();
  const reader = new FileReader();
  reader.onload = () => {
    // Strip the data: URL prefix; the host wants raw base64.
    const s = String(reader.result || '');
    const comma = s.indexOf(',');
    api.postMessage({ kind: 'attach', id, name: file.name || 'image.png',
                      data: comma >= 0 ? s.slice(comma + 1) : '' });
  };
  reader.onerror = () => {
    const a = attached.get(id);
    if (a) { a.error = 'could not be read'; renderAttachments(); }
  };
  reader.readAsDataURL(file);
}

// Depth-counted, not a boolean: dragenter/dragleave fire for every child element the
// cursor crosses, so a naive toggle flickers the veil off the moment the pointer moves
// over the composer inside the drop zone.
let dragDepth = 0;
const veil = () => $('dropVeil');
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  if (++dragDepth === 1) veil().classList.add('on');
});
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('dragleave', (e) => {
  e.preventDefault();
  if (--dragDepth <= 0) { dragDepth = 0; veil().classList.remove('on'); }
});
window.addEventListener('drop', (e) => {
  e.preventDefault();
  dragDepth = 0;
  veil().classList.remove('on');
  const dt = e.dataTransfer;
  if (!dt) return;
  for (const f of Array.from(dt.files || [])) offerFile(f);
});
box.addEventListener('paste', (e) => {
  const items = (e.clipboardData || {}).items || [];
  for (const it of Array.from(items)) {
    if (it.kind === 'file') {
      const f = it.getAsFile();
      if (f) { e.preventDefault(); offerFile(f); }
    }
  }
});

const submit = () => {
  const text = box.value.trim();
  // A message that is ONLY images is legitimate -- dropping a screenshot and pressing
  // send means "look at this" -- so the guard is "nothing at all", not "no text".
  const ready = [...attached.values()].filter((a) => a.path).map((a) => a.path);
  if (!text && ready.length === 0) return;
  api.postMessage({ kind: 'message', text: text || 'Look at this.', images: ready });
  attached.clear();
  renderAttachments();
  box.value = '';
  box.style.height = 'auto';
};
$('send').onclick = submit;

// Stop. The sidecar's cancel is the violent interrupt -- it sets the token mid-stream
// rather than waiting for a turn boundary, which is the whole point of a stop button.
// Disabled the instant it is pressed so a second click cannot queue a second cancel.
$('stop').onclick = () => {
  $('stop').disabled = true;
  busy(true, 'Stopping…', 'WAITING');
  api.postMessage({ kind: 'cancel' });
};

// --- context meter --------------------------------------------------------
// Bands, not a gradient: the number only matters near the top, and a bar that is already
// amber at 40% trains you to ignore it.
let lastCompactions = 0;
function paintContext(used, max, compactions) {
  const ctx = $('ctx');
  if (!max) { ctx.classList.add('idle'); return; }
  ctx.classList.remove('idle');
  const pct = Math.max(0, Math.min(100, (used / max) * 100));
  $('ctxFill').style.width = pct.toFixed(1) + '%';
  ctx.classList.toggle('warm', pct >= 70 && pct < 88);
  ctx.classList.toggle('hot', pct >= 88);
  $('ctxLabel').textContent = Math.round(pct) + '% of context';

  const chip = $('ctxCompact');
  if (compactions > 0) {
    chip.classList.add('on');
    chip.textContent = 'compacted ' + compactions + '×';
    // Pulse only on a NEW one. Re-running the animation every turn would make a steady
    // state look like a recurring event.
    if (compactions > lastCompactions) {
      chip.classList.remove('bump');
      void chip.offsetWidth;
      chip.classList.add('bump');
    }
  } else {
    chip.classList.remove('on');
  }
  lastCompactions = compactions;
}

// --- run history ----------------------------------------------------------
function paintHistory(runs) {
  const el = $('history');
  el.textContent = '';
  if (!runs || runs.length === 0) {
    const p = document.createElement('div');
    p.className = 'empty';
    p.textContent = 'No runs yet in this workspace.';
    el.append(p);
    return;
  }
  for (const r of runs) {
    const row = document.createElement('div');
    row.className = 'run' + (r.resumable ? ' resumable' : '');
    const dot = document.createElement('span');
    // Three states, not two. A run that never reached an ending is neither a success nor
    // a failure -- it is the one most likely worth reopening, so it must not wear the
    // same dot as a run that finished badly.
    dot.className = 'rdot ' + (r.finished === false ? 'dead' : (r.completed ? 'ok' : 'no'));
    const m = document.createElement('span');
    m.className = 'rmission';
    m.textContent = r.mission;
    // \\n, not \\\\n: this source is a TEMPLATE LITERAL, so an escape written once is
    // resolved by TypeScript and reaches the browser as a REAL newline -- which splits the
    // string literal across two lines and kills the entire view script at parse time.
    // tsc cannot see it: to the compiler this whole function is one string.
    m.title = r.mission + '\\n' + r.reason +
              (r.resumable ? '\\n\\nClick to reopen this conversation.'
                           : '\\n\\nNothing recorded to reopen.');
    const meta = document.createElement('span');
    meta.className = 'rmeta';
    meta.textContent = (r.finished === false ? 'died · ' : '') +
                       r.iterations + ' turns · ' + when(r.at);
    row.append(dot, m, meta);
    // Only a row with something behind it is clickable. A hover state on a row that
    // cannot open is a promise the panel cannot keep.
    if (r.resumable && r.runId) {
      row.onclick = () => {
        api.postMessage({ kind: 'resume', runId: r.runId });
        $('history').classList.remove('open');
        $('histBtn').classList.remove('on');
      };
    }
    el.append(row);
  }
}

// Relative, because "3 minutes ago" is what you want from a history list and an ISO
// timestamp is what you want from a log.
function when(ms) {
  const s = Math.max(0, (Date.now() - ms) / 1000);
  if (s < 90) return 'just now';
  const m = s / 60;
  if (m < 60) return Math.round(m) + 'm ago';
  const h = m / 60;
  if (h < 24) return Math.round(h) + 'h ago';
  return Math.round(h / 24) + 'd ago';
}

$('histBtn').onclick = () => {
  const el = $('history');
  const open = el.classList.toggle('open');
  $('histBtn').classList.toggle('on', open);
  if (open) {
    $('drawer').classList.remove('open');
    $('gear').classList.remove('on');
    api.postMessage({ kind: 'history' });
  }
};

// --- the model bar --------------------------------------------------------
// Four states, one button, and the button says what pressing it does rather than what
// the state is -- the dot and the name already say that.
const MODEL_ACTION = {
  unloaded: 'Load', loading: 'Loading…', ready: 'Unload', failed: 'Retry',
};
let modelState = 'unloaded';
/** Whether the status line currently belongs to a load, and must be handed back. */
let loadingModel = false;

function paintModel(m) {
  modelState = m.state;
  $('modelDot').className = 'dot ' + (
    m.state === 'ready' ? 'ok' : m.state === 'failed' ? 'failed' : m.state);
  const name = m.model_dir ? m.model_dir.split('/').filter(Boolean).pop() : '';
  $('modelName').textContent =
    m.state === 'ready' ? name + (m.elapsed_ms > 0 ? ' · loaded in ' + (m.elapsed_ms / 1000).toFixed(1) + 's' : '')
    : m.state === 'loading' ? 'Loading ' + name + '…'
    : m.state === 'failed' ? 'Failed: ' + name
    : 'No model loaded';
  $('modelName').title = m.model_dir || '';
  const act = $('modelAction');
  act.textContent = MODEL_ACTION[m.state] || 'Load';
  act.disabled = m.state === 'loading';
}

$('modelAction').onclick = () => {
  if (modelState === 'loading') return;
  api.postMessage({ kind: modelState === 'ready' ? 'unload_model' : 'load_model' });
};
$('modelPick').onclick = () => api.postMessage({ kind: 'pick_model' });
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
// A property of the LOADED CHECKPOINT, not a setting -- it arrives on model_status and is
// never written back. Starts false so the control reads as unavailable until a model has
// actually said otherwise; claiming the level works and finding out later that it never
// reached the prompt is the failure this whole flag exists to prevent.
let effortSupported = false;

const put = (key, value) => {
  settings[key] = value;
  api.postMessage({ kind: 'setting', key, value });
};

// See viewBuildId(): the one thing on screen that says which VIEW code is running, as
// against which sidecar. They update on different schedules and the event log only knows
// about the sidecar.
$('viewBuild').textContent = VIEW_BUILD;

// One panel at a time: two stacked disclosures push the composer off a narrow sidebar.
$('gear').onclick = () => {
  const open = $('drawer').classList.toggle('open');
  $('gear').classList.toggle('on', open);
  if (open) {
    $('history').classList.remove('open');
    $('histBtn').classList.remove('on');
  }
};

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
seg('segEffort', 'reasoningEffort', String);

function sw(id, key) {
  $(id).onclick = () => { put(key, !settings[key]); paint(); };
}
sw('swExec', 'autoApproveExec');
sw('swWrite', 'autoApproveWrites');
$('swSpec').onclick = () => {
  if (settings.checkpointKind !== 'dense') return;
  put('speculativeDecoding', !settings.speculativeDecoding);
  paint();
};

// --- the thinking toggle --------------------------------------------------
// Applied to the disclosures ALREADY IN THE FEED as well as the ones still to come, which
// is the whole difference between a toggle and a default.
//
// Before this, every turn's reasoning arrived in a fresh <details open=false>. You opened
// one, read it, and the next turn produced another shut one -- with its summary already
// rewritten to "Thought for 1.7s", so there was not even a "Thinking…" to notice. Wanting
// to watch the reasoning meant re-opening a disclosure on every single turn, and the
// setting that would have said "yes, always" did not exist.
function setThinking(on) {
  showThinking = on;
  $('thinkToggle').classList.toggle('on', on);
  for (const d of feed.querySelectorAll('details.thought')) d.open = on;
}
$('thinkToggle').onclick = () => {
  setThinking(!showThinking);
  // Through settings, not a local flag: the webview is destroyed whenever the panel is
  // hidden, and a preference that resets every time you look away is not a preference.
  put('showThinking', showThinking);
};

// --- the three budgets ----------------------------------------------------
// All three, on one panel, with the consequence spelled out. A turn limit alone is a trap:
// raise it without the clock and the run stops at the same wall it always did, for a
// different stated reason.
//
// The seconds-per-turn figure is measured, and it had to be re-measured. 12.5 s came off
// the mission the original ceiling was set from, on a text-only checkpoint with a roomy
// context budget. A vision-enabled 27B compacting every ~5 turns costs far more, because
// every compaction buys one full re-prefill: 73 turns of a 2026-08-17 run took 103 min of
// model time, 85 s of it per turn, 40% of that in prefill alone. The drawer would rather
// be right about the slow case, since the slow case is the one where the clock binds.
const SECONDS_PER_TURN = 60;

const dragging = (id, key, live) => {
  const r = $(id);
  r.oninput = () => { live(Number(r.value)); repaintBudget(); };
  r.onchange = () => put(key, Number(r.value));
};
const repaintBudget = () => paintBudget(Number($('turnRange').value),
                                        Number($('clockRange').value),
                                        Number($('stallRange').value));
dragging('turnRange', 'maxIterations', (v) => { $('turnVal').textContent = v + ' turns'; });
dragging('clockRange', 'wallClockSeconds', (v) => { $('clockVal').textContent = Math.round(v / 60) + ' min'; });
dragging('stallRange', 'stallSeconds', (v) => { $('stallVal').textContent = Math.round(v / 60) + ' min'; });

// The longest single turn that is not a fault: worst measured time-to-first-token (~129 s)
// plus a full-length generation at the measured decode rate plus a shell command riding
// its own 300 s limit. A stall dial under this stops runs that are working.
const LONGEST_HONEST_TURN_S = 700;

function paintBudget(turns, seconds, stall) {
  const warn = $('budgetWarn');
  // Ordered by which mistake costs the most. A stall dial set below one honest turn kills
  // healthy runs at random, which is worse than a clock that binds early, which is worse
  // than the intended state.
  if (stall < LONGEST_HONEST_TURN_S) {
    warn.className = 'warnbox';
    warn.textContent =
      'Stuck-limit is under one slow turn: a single generation plus a shell command can '
      + 'honestly take ~' + Math.round(LONGEST_HONEST_TURN_S / 60) + ' min, so '
      + Math.round(stall / 60) + ' min will stop runs that are still working. '
      + 'Raise it, or healthy turns will read as hangs.';
    return;
  }
  const clockStopsAt = Math.floor(seconds / SECONDS_PER_TURN);
  if (clockStopsAt < turns) {
    warn.className = 'warnbox';
    warn.textContent =
      'The clock stops this run first: at ~' + SECONDS_PER_TURN + 's a turn, ' +
      Math.round(seconds / 60) + ' min is about ' + clockStopsAt + ' turns, not ' + turns +
      '. Raise the time limit or the extra turns are unreachable.';
    return;
  }
  warn.className = 'warnbox calm';
  warn.textContent =
    'Turns run out first, which is the intent: ~' + Math.round(turns * SECONDS_PER_TURN / 60) +
    ' min of work at the measured rate. A run that hangs is stopped after ' +
    Math.round(stall / 60) + ' min without a turn, and the clock is only the backstop.';
}

$('promptBox').addEventListener('change', () => {
  put('prompts.' + (settings.mode || 'agent'), $('promptBox').value);
});

// --- the operator's check -------------------------------------------------
// The ONLY verification the harness runs, and the only one it can run: it is the
// operator's command, executed verbatim after any turn that writes, with its output put
// in front of the model. Empty means the run ends purely on the model's own say-so, so
// the drawer says which of the two claims "Complete" is about to make.
$('checkBox').addEventListener('change', () => {
  put('verifyContract', $('checkBox').value.trim());
  paint();
});

function paintCheck() {
  const cmd = (settings.verifyContract || '').trim();
  $('checkBox').value = cmd;
  $('checkState').textContent = cmd ? '· on' : '· off';
  const warn = $('checkWarn');
  warn.className = cmd ? 'warnbox calm' : 'warnbox';
  warn.textContent = cmd
    ? 'Runs after any turn that writes a file. Its output goes to the agent, and "Complete" means this passed.'
    : 'No check: the run ends when the agent says it is done, and nothing verifies that. Set a build or test command.';
}

function shortModelName(dir) {
  const base = (dir || '').split('/').filter(Boolean).pop() || dir || '';
  return base.replace(/-MLX-4bit$/i, '').replace(/-MLX.*$/i, '') || base;
}

function paintModelSwitch() {
  const host = $('modelSwitch');
  const current = settings.modelDir || '';
  const recents = Array.isArray(settings.recentModelDirs) ? settings.recentModelDirs : [];
  const other = recents.find((d) => d && d !== current);
  host.textContent = '';
  if (!current || !other) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  for (const dir of [current, other]) {
    const b = document.createElement('button');
    b.textContent = shortModelName(dir);
    b.title = dir;
    b.classList.toggle('on', dir === current);
    b.onclick = () => {
      if (dir === current || modelState === 'loading') return;
      api.postMessage({ kind: 'switch_model', dir });
    };
    host.append(b);
  }
}

function paintSpec() {
  const dense = settings.checkpointKind === 'dense';
  $('specRow').classList.toggle('off', !dense);
  $('swSpec').classList.toggle('on', settings.speculativeDecoding === true);
  const warn = $('specWarn');
  if (settings.checkpointKind !== 'dense') {
    warn.className = 'warnbox';
    warn.textContent = settings.checkpointKind === 'moe'
      ? 'MTP speculation only runs on the dense 27B. The setting is kept; switching back to dense restores it. A3B is never sent the draft head.'
      : 'Available on the dense 27B. A3B is never sent the MTP draft head, so switching to it cannot fail the load.';
    return;
  }
  if (settings.speculativeDecoding && !settings.hasUsableDraft) {
    warn.className = 'warnbox';
    warn.textContent =
      'On, but no MTP draft head is configured. Use Change and pick the dense checkpoint again — a sibling *-MTP-4bit folder is detected automatically.';
    return;
  }
  warn.className = settings.speculativeDecoding ? 'warnbox calm' : 'warnbox';
  warn.textContent = settings.speculativeDecoding
    ? 'Draft tokens from the MTP head. Reloads the dense model when you toggle this.'
    : 'Off: the dense model loads without the MTP head.';
}

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
  paintSpec();
  paintModelSwitch();
  // Thinking level. Greyed rather than hidden when the checkpoint has no notion of it:
  // the setting is real and shared across checkpoints, so the honest thing is to show
  // the value it holds and say why it is doing nothing here -- a control that vanishes
  // reads as a missing feature, and one that looks live reads as a lie.
  $('segEffort').querySelectorAll('button').forEach((b) => {
    b.classList.toggle('on', b.dataset.v === settings.reasoningEffort);
    b.disabled = !effortSupported;
  });
  $('segEffort').classList.toggle('off', !effortSupported);
  $('effortState').textContent = effortSupported ? '' : 'not supported here';
  $('effortWarn').textContent = !effortSupported
    ? 'This checkpoint has no thinking levels — the setting is kept and ignored. Qwen3.8 has them; Qwen3.6 does not.'
    : settings.reasoningEffort === 'low'
    ? 'Briefest reasoning. Good for mechanical work, worse for anything needing a plan.'
    : '';
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
  // The budgets. A range input silently CLAMPS a value outside its bounds, and this one
  // writes back on change -- so a settings.json saying 1500 turns would show as 600 and
  // become 600 the moment the slider was touched. Widen the track instead: the drawer
  // must never quietly overrule a number the operator set somewhere else.
  const budget = (id, key, fallback, span, fmt) => {
    const v = typeof settings[key] === 'number' ? settings[key] : fallback;
    const r = $(id);
    r.max = String(Math.max(span, v));
    r.value = v;
    $(fmt.id).textContent = fmt.text(v);
    return v;
  };
  const turns = budget('turnRange', 'maxIterations', 200, 600,
    { id: 'turnVal', text: (v) => v + ' turns' });
  const seconds = budget('clockRange', 'wallClockSeconds', 14400, 43200,
    { id: 'clockVal', text: (v) => Math.round(v / 60) + ' min' });
  const stall = budget('stallRange', 'stallSeconds', 1200, 7200,
    { id: 'stallVal', text: (v) => Math.round(v / 60) + ' min' });
  paintBudget(turns, seconds, stall);

  paintCheck();

  $('promptMode').textContent = '· ' + (settings.mode || 'agent');
  $('promptBox').value = settings['prompts.' + (settings.mode || 'agent')] || '';
}

// --- tool arguments -------------------------------------------------------
//
// tool_args is a JSON object of every parameter, keyed by name. It was the first
// parameter's bare text until the sidecar started sending all of them, which is why
// parseToolArgs tolerates a non-JSON string instead of throwing: an older sidecar against
// a newer view should degrade to "no named arguments", not to a broken turn row.
function parseToolArgs(raw) {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch (_) {
    return {};
  }
}

// What the collapsed tool row shows next to the tool name. The row has one line for this,
// so it names the thing acted ON -- the path, the command -- and never the whole argument
// object, which is what printing raw JSON here would do.
function argsPreview(argsObj, raw) {
  const first = ['path', 'command', 'pattern', 'text', 'query', 'subdir', 'question', 'plan']
    .map((k) => argsObj[k])
    .find((v) => typeof v === 'string' && v.trim());
  const shown = first !== undefined ? first
    : (Object.keys(argsObj).length === 0 && typeof raw === 'string' ? raw : '');
  const line = String(shown).split('\\n')[0].trim();
  return line.length > 120 ? line.slice(0, 119) + '…' : line;
}

const Q_BULLET = /^\\s*[-*•]\\s+/;
const Q_INDENT = /^\\s+\\S/;
const Q_MARKER = /(?:Option\\s+[A-Za-z0-9]+[:\\)]|[-*•]\\s+|\\d+[\\.\\)]\\s+)/i;
const Q_LEAD_MARKER = /^(?:Option\\s+[A-Za-z0-9]+\\s*[:\\)]\\s*|[-*•]\\s+|\\d+\\s*[\\.\\)]\\s+|[A-Za-z]\\s*[\\.\\)]\\s+)/i;

// One block of text -> one option. The first line names it; anything under it is detail,
// which the card shows but never makes separately clickable.
// Markdown emphasis is not part of the choice. A model writing an enumerated list in
// prose bolds the name of each option -- "1. **Katana** (Traditional Japanese...)" -- and
// the label goes into the button as textContent AND back to the model as the answer, so
// the asterisks would be visible in both. Stripped here rather than at the call sites,
// because both paths produce labels and only one of them used to be reachable.
function stripEmphasis(s) {
  return s.replace(/\\*\\*/g, '').replace(/__/g, '').trim();
}

function questionBlockToOption(block) {
  const lines = block.split(/\\n/).map((s) => s.trim()).filter(Boolean);
  const label = stripEmphasis((lines.shift() || '').replace(Q_LEAD_MARKER, '').trim());
  const detail = lines
    .map((l) => stripEmphasis(l.replace(Q_BULLET, '').trim()))
    .filter(Boolean)
    .join(' · ');
  return { label: label, detail: detail };
}

// The selectable options for a question card, from the 'options' argument alone.
//
// It deliberately does NOT fall back to the turn's summary. That summary is the loop's own
// status string -- "asked the operator; the run stops here" -- and the old fallback turned
// it into a card titled with it, offering it as the single thing to click. A question that
// arrives without options is a malformed question, and the honest render is the plain tool
// row, not one fabricated choice.
//
// EVERY LINE WAS AN OPTION until 2026-08-24. The tool description asks for "one per line"
// and the model very reasonably writes richer than that -- measured, from real runs:
//
//     Keep the current layout
//       - Familiar to existing users
//       - Fewer moving parts
//
//     Switch to a split view
//       - Side-by-side comparison
//
// Splitting that on /\\n+/ collapses the blank line that separates the two real choices and
// offers EIGHT clickable rows, six of them fragments of the argument for one of the other
// two. The delimiters a model actually uses, in descending order of how reliably each one
// means "next option": a blank line, then a line that is neither indented nor bulleted,
// then markers inside a single line.
function questionOptions(argsObj) {
  if (Array.isArray(argsObj.options)) {
    return argsObj.options
      .map((s) => String(s).trim())
      .filter(Boolean)
      .map((label) => ({ label: label, detail: '' }));
  }
  if (typeof argsObj.options !== 'string' || !argsObj.options.trim()) {
    return [];
  }
  // Real newlines, or the two-character escape "\\n" that survives a round trip as text.
  const normalized = argsObj.options
    .replace(/\\\\r\\\\n/g, '\\n')
    .replace(/\\\\n/g, '\\n')
    .replace(/\\r\\n/g, '\\n');

  // 1. A blank line is unambiguous, and the only delimiter that still reads correctly when
  //    an option carries detail lines of its own.
  const blocks = normalized.split(/\\n\\s*\\n/).map((s) => s.trim()).filter(Boolean);
  if (blocks.length > 1) {
    return blocks.map(questionBlockToOption).filter((o) => o.label);
  }

  const lines = normalized.split(/\\n/).filter((l) => l.trim());

  // 2. No blank lines: a line that is neither indented nor bulleted starts an option, and
  //    indented or bulleted lines belong to the one above it. Guarded on there BEING such
  //    a line, because a flat "- a / - b / - c" list is bulleted all the way down and
  //    those bullets are the options themselves, not detail.
  const hasHeaderLine = lines.some((l) => !Q_INDENT.test(l) && !Q_BULLET.test(l));
  if (lines.length > 1 && hasHeaderLine) {
    const groups = [];
    for (const line of lines) {
      const isDetail = Q_INDENT.test(line) || Q_BULLET.test(line);
      if (!isDetail || groups.length === 0) {
        groups.push([line]);
      } else {
        groups[groups.length - 1].push(line);
      }
    }
    return groups
      .map((g) => questionBlockToOption(g.join('\\n')))
      .filter((o) => o.label);
  }

  // 3. Everything on one line, with the markers inside it.
  if (lines.length === 1 && Q_MARKER.test(lines[0])) {
    const split = lines[0]
      .split(/(?=(?:Option\\s+[A-Za-z0-9]+[:\\)]|[-*•]\\s+|\\d+[\\.\\)]\\s+))/i)
      .map((s) => s.trim())
      .filter(Boolean);
    if (split.length > 1) {
      return split.map(questionBlockToOption).filter((o) => o.label);
    }
  }

  // 4. Plain one-per-line, including the flat bulleted list from (2).
  return lines.map((l) => questionBlockToOption(l)).filter((o) => o.label);
}

// An enumerated choice list written INSIDE the question text, with no 'options' argument.
//
// WHY THIS EXISTS. 'ask_user' takes only 'question', so it can never carry options and
// never drew a card -- the run just yielded to the composer box. But the model uses both
// tools for the same job: on 2026-08-24 it asked a four-way design question through
// 'ask_user' and wrote the choices into the question itself --
//
//     Which approach should we take for the shared helper?
//
//     Option 1: Extract it now (one PR, larger diff)
//     Option 2: Patch the call site (smallest change)
//     ...
//
// -- which is a perfectly good four-option question that rendered as a plain tool row and
// a text box. Whether the card appears should not depend on which of two tools the model
// reached for, so the options are read from where it actually put them.
//
// STRONG MARKERS ONLY, and at least two of them. A bulleted list inside prose is usually
// not a choice set, and inventing a card around one would be the same mistake as the old
// fallback that offered the loop's status string as the single thing to click. The
// single-letter form requires exactly one character before the punctuation, so "So. Then
// I..." is not an option and "A) ..." is.
const Q_ENUM_LINE = /^(?:option\\s+[0-9a-z]+\\s*[:.\\)]|[0-9]{1,2}\\s*[.\\)]\\s|[a-z]\\s*[.\\)]\\s)/i;

function questionFromText(text) {
  const empty = { question: text, options: [] };
  if (typeof text !== 'string' || !text.trim()) {
    return empty;
  }
  const lines = text.replace(/\\r\\n/g, '\\n').split(/\\n/);
  const firstMarker = lines.findIndex((l) => Q_ENUM_LINE.test(l.trim()));
  if (firstMarker < 0) {
    return empty;
  }
  const markerCount = lines.filter((l) => Q_ENUM_LINE.test(l.trim())).length;
  if (markerCount < 2) {
    return empty;
  }

  // Everything above the first marker is the question being asked. Blank-line padding
  // between it and the list is presentation, not content.
  const head = lines.slice(0, firstMarker).join(' ').trim();

  // A marker line opens an option; anything after it that is not a marker is its detail,
  // which is how a model writes a choice that needs a sentence of justification.
  const groups = [];
  for (const line of lines.slice(firstMarker)) {
    if (!line.trim()) {
      continue;
    }
    if (Q_ENUM_LINE.test(line.trim()) || groups.length === 0) {
      groups.push([line.trim()]);
    } else {
      groups[groups.length - 1].push(line.trim());
    }
  }
  const options = groups
    .map((g) => questionBlockToOption(g.join('\\n')))
    .filter((o) => o.label);
  if (options.length < 2) {
    return empty;
  }
  return { question: head || text, options: options };
}

// THE WHOLE DECISION: given the call's arguments, what does the card offer?
//
// Named and placed here, above the inbound marker, because scripts/verify-question-options.js
// slices this region out of the file and runs it. The gate used to live inline in the
// render block where no harness could reach it -- and that is exactly where the bug was.
//
// Returns fewer than two options when there is no real choice to offer. The caller must
// not draw a card for that: see the render site.
function questionCard(argsObj, questionText) {
  let options = questionOptions(argsObj);
  // ONE OPTION IS NOT A CHOICE. This used to be 'options.length === 0', so a single
  // parsed option locked the card to it and the question text was never consulted.
  //
  // Measured, verbatim from the run that exposed it. The model put the real choices in
  // the question, numbered, one per line -- and passed 'options' as a one-line SUMMARY
  // of them, "Katana / Tachi / Dual Swords / Long Sword". questionOptions reads that as
  // one option, correctly: no newline and no marker, so nothing in it says "four
  // choices". One beat zero, the fallback was skipped, and a four-way design question
  // reached the human as a single button labelled with every answer at once.
  //
  // The question text wins when it holds two or more enumerated options: a model that
  // writes them out in full there and abbreviates them in 'options' has said which of
  // the two it meant.
  if (options.length < 2) {
    const embedded = questionFromText(questionText);
    if (embedded.options.length > 1) {
      return { question: embedded.question, options: embedded.options };
    }
  }
  return { question: questionText, options: options };
}

// --- inbound --------------------------------------------------------------
window.addEventListener('message', (e) => {
  const { kind, payload } = e.data;

  if (kind === 'settings') {
    settings = payload;
    setThinking(payload.showThinking === true);
    paint();
  }

  if (kind === 'model') {
    paintModel(payload);
    effortSupported = payload.supports_reasoning_effort === true;
    paint();
    // A load owns the sidecar's only thread, so nothing else can be happening while one
    // runs -- which is both why "Loading the model" is safe to put in the status line and
    // why a run can never be in flight underneath it. Saying it is the difference between
    // a wait with a reason and the 'Thinking' that used to sit there over a process
    // holding no weights at all.
    //
    // loadingModel is what hands the line back. Without it the status kept whichever
    // word the LOAD had put there once the load was over -- "Loading the model", with
    // the model long since loaded. Exactly the stale-status bug this whole change is
    // about, reintroduced one layer up.
    if (payload.state === 'loading') { loadingModel = true; busy(true, 'Loading the model', 'THINKING'); }
    else if (payload.state === 'failed') { loadingModel = false; finish('Model failed to load', 'FAILED'); }
    else if (loadingModel) { loadingModel = false; busy(false, 'Idle', 'IDLE'); }
  }

  if (kind === 'error') {
    closeBubble();
    const d = document.createElement('div');
    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = 'LM_Pipe';
    const body = document.createElement('div');
    body.textContent = payload.text;
    d.append(who, body);
    add(d, 'error');
  }

  if (kind === 'run_start') {
    lastTurnTool = '';
    lastQuestionCardDrawn = false;
    $('mission').textContent = payload.mission;
    // Only a DELIBERATE start over clears the feed. When the mission came from the
    // composer the user's message is already on screen, and wiping it to announce the
    // run it started would delete the thing they just typed.
    if (payload.reset) {
      // The .msg children only. feed.textContent = '' would take the live row with them
      // -- and with it the orb's WebGL context, which does not come back.
      for (const el of [...feed.querySelectorAll('.msg')]) el.remove();
      $('plan').textContent = ''; $('perf').textContent = '';
    }
    closeBubble();
    // What this run may actually do, from the run itself rather than from the segmented
    // control -- the control describes the NEXT run, and the two diverge exactly when it
    // matters most, which is when someone has just changed it.
    $('modeNow').textContent = payload.mode || '';
    busy(true, 'Thinking', 'THINKING');
  }

  // The host finished writing a dropped image. The chip stops being pending and gains the
  // path the sidecar will be given; a failure keeps the chip and says why, rather than
  // vanishing and leaving the user unsure whether the drop registered at all.
  if (kind === 'attached') {
    const a = attached.get(msg.id);
    if (a) {
      if (msg.error) { a.error = msg.error; } else { a.path = msg.path; }
      renderAttachments();
    }
    return;
  }
  if (kind === 'said') {
    // The user has just typed. Whatever the agent was still saying is now the PAST, and
    // letting it trickle in under a message the human has already sent reads as the reply
    // arriving before the question.
    closeBubble();
    flushQueue();
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
      openThought();
      renderMd(thoughtMd, thoughtMd.stream.feed(payload.text));
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
    lastTurnTool = payload.tool_name || '';
    lastQuestionCardDrawn = false;
    const isQuestionTool = payload.tool_name === 'ask_question' || payload.tool_name === 'ask_user';
    if (isQuestionTool) {
      try {
        const argsObj = parseToolArgs(payload.tool_args);
        let questionText = argsObj.question || 'Clarification required:';
        const card = questionCard(argsObj, questionText);
        questionText = card.question;
        const optionList = card.options;

        // NO CARD FOR A SINGLE OPTION. A one-button card looks like a choice, is not one,
        // and sends an answer the human never chose. The honest render for a question we
        // could not find a choice set in is the plain tool row and the composer box --
        // the same conclusion questionOptions' own comment reached about fabricating a
        // choice out of the loop's status string.
        if (optionList.length >= 2) {
          lastQuestionCardDrawn = true;
          const qCard = document.createElement('div');
          qCard.className = 'question-card';

          const qHeader = document.createElement('div');
          qHeader.className = 'q-header';

          const qTitle = document.createElement('h4');
          qTitle.textContent = questionText;

          const qSub = document.createElement('div');
          qSub.className = 'q-subtitle';
          qSub.textContent = 'Select option(s) to proceed:';

          qHeader.append(qTitle, qSub);

          const optsContainer = document.createElement('div');
          optsContainer.className = 'q-options';

          const selectedIndices = new Set();

          const submitBtn = document.createElement('button');
          submitBtn.className = 'q-submit-btn';
          submitBtn.style.display = 'none';
          submitBtn.textContent = 'Submit Selection';

          const submitSelection = () => {
            const chosenTexts = Array.from(selectedIndices)
              .sort((a, b) => a - b)
              .map(i => optionList[i].label);
            if (chosenTexts.length === 0) return;
            const textToSubmit = chosenTexts.length === 1 ? chosenTexts[0] : chosenTexts.join('\\n\\n');

            api.postMessage({ kind: 'message', text: textToSubmit });
            qCard.classList.add('submitted');
            qCard.style.opacity = '0.7';
            qCard.style.pointerEvents = 'none';
            submitBtn.style.display = 'none';
          };

          submitBtn.onclick = submitSelection;

          optionList.forEach((opt, idx) => {
            const btn = document.createElement('button');
            btn.className = 'q-opt-btn';
            btn.type = 'button';

            const badge = document.createElement('span');
            badge.className = 'q-opt-badge';
            const letterBadge = String.fromCharCode(65 + idx);
            badge.textContent = letterBadge;

            // The label is the choice; the detail is the case FOR it. Both belong on the
            // card -- the model wrote the detail to be read before choosing -- but only
            // the label is the answer, so only the label is sent back.
            const txt = document.createElement('span');
            txt.className = 'q-opt-text';
            const labelEl = document.createElement('span');
            labelEl.className = 'q-opt-label';
            labelEl.textContent = opt.label;
            txt.append(labelEl);
            if (opt.detail) {
              const detailEl = document.createElement('span');
              detailEl.className = 'q-opt-detail';
              detailEl.textContent = opt.detail;
              txt.append(detailEl);
            }

            btn.append(badge, txt);

            btn.onclick = () => {
              if (selectedIndices.has(idx)) {
                selectedIndices.delete(idx);
                btn.classList.remove('selected');
                badge.textContent = letterBadge;
              } else {
                selectedIndices.add(idx);
                btn.classList.add('selected');
                badge.textContent = '✓';
              }

              if (selectedIndices.size > 1) {
                submitBtn.style.display = 'inline-block';
                submitBtn.textContent = 'Submit Selection (' + selectedIndices.size + ')';
              } else if (selectedIndices.size === 1) {
                // Submit immediately on single choice click or show submit button
                submitSelection();
              } else {
                submitBtn.style.display = 'none';
              }
            };

            optsContainer.append(btn);
          });

          const footer = document.createElement('div');
          footer.className = 'q-footer';
          footer.append(submitBtn);

          qCard.append(qHeader, optsContainer, footer);
          add(qCard, '');
          busy(true, 'Waiting for choice', 'WAITING');
          if (window.__orb) window.__orb.impulse('tool');
          return;
        }
      } catch (err) {
        // fallback to standard tool row on error
      }
    }

    // A TURN WITH NO TOOL IS NOT A TOOL ROW. TextOnly still went through on_turn with an
    // empty name and the default ToolResult status (ToolError), so the view drew a red
    // "TextOnly" details element under the answer. Measured on r-18cf3831ea73c5a8-2d4f679d
    // turn 6 -- the "**Fixed.**" paragraph -- immediately before finish: that row is the
    // "malformed output at the very end right before the last turn".
    if (!payload.tool_name) {
      busy(true, 'Thinking', 'THINKING');
      return;
    }

    const d = document.createElement('details');
    d.className = 'tool';
    const s = document.createElement('summary');
    const dot = document.createElement('span');
    dot.className = 'dot ' + (payload.tool_status === 'ok' ? 'ok'
      : (payload.tool_status === 'refused' || payload.tool_status === 'denied') ? 'refused' : 'failed');
    const n = document.createElement('span'); n.className = 'name'; n.textContent = payload.tool_name || payload.outcome;
    const a = document.createElement('span'); a.className = 'args';
    a.textContent = argsPreview(parseToolArgs(payload.tool_args), payload.tool_args);
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
    // A CHECKLIST THE READER FOLDED AWAY STAYS FOLDED. The card is rebuilt from scratch on
    // every \`plan\` call, and \`open = true\` was hardcoded into the rebuild -- so collapsing
    // it bought nothing: the next thing the agent did sprang it back open, and on a run
    // that restates its checklist every few turns that is every few turns.
    //
    // Read before the wipe, because the wipe is what destroys the state being read. Absent
    // (first render of a run) means open, which is where a new checklist should start.
    const prev = $('plan').querySelector('details');
    const wasOpen = prev ? prev.open : true;
    $('plan').innerHTML = '';
    const det = document.createElement('details');
    det.open = wasOpen;
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
    // A check that never EXECUTED said nothing about the workspace, and rendering it as
    // FAIL sends the reader to the code when the broken thing is the command.
    const ran = payload.ran !== false;
    p.className = 'pill ' + (!ran ? 'fail' : payload.passed ? 'ok' : 'fail');
    p.textContent = !ran ? 'COULD NOT RUN' : payload.passed ? 'PASS' : 'FAIL';
    const c = document.createElement('code');
    c.textContent = payload.contract;
    d.append(p, c);
    add(d, '');
    busy(true, payload.passed ? 'Check passed' : 'Check failed',
         payload.passed ? 'DONE' : 'FAILED');
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
    const answer = (ok, remember, allowWrites, allowForRun) => {
      api.postMessage({
        kind: 'approve', id: payload.request_id, approved: ok, remember,
        allowWrites: allowWrites === true, allowForRun: allowForRun === true,
      });
      card.remove();
    };
    yes.onclick = () => answer(true);
    no.onclick = () => answer(false);
    row.append(yes, no);

    // The WRITE gate's counterpart to "Always allow", and it is scoped to the run rather
    // than remembered, because the two gates repeat differently. A command recurs
    // verbatim, so a stored rule pays off on every later turn. A path does not: a run
    // rewriting eleven scaffolded stubs writes eleven different paths once each, so a
    // per-path rule could only ever be written after its one use and the operator would
    // still answer eleven cards. MEASURED: exactly eleven, in one run, with
    // auto-approve-writes already on.
    //
    // A write card is the one with no command -- the same test the sidecar's approver
    // uses, so the button appears exactly where the backend will honour it.
    if (!payload.command) {
      const runAllow = document.createElement('button');
      runAllow.className = 'ghost';
      runAllow.textContent = 'Allow writes for this run';
      runAllow.title =
        'Stop asking about whole-file writes for the rest of this run. Not remembered: ' +
        'the next run asks again.';
      runAllow.onclick = () => answer(true, undefined, true);
      row.append(runAllow);
    }

    // Run-scoped consent for a COMMAND, the counterpart to "Allow writes for this run".
    // Offered on every command card that is not irreversible, including the ones no stored
    // rule could ever match -- which is most of what a model actually writes, because it
    // composes a fresh "VAR=...; $VAR sub --flag | head" every turn. The sidecar latches it
    // and honours it on the next turn of THIS run; nothing is persisted.
    if (payload.command && !payload.irreversible) {
      const runAllow = document.createElement('button');
      runAllow.className = 'ghost';
      runAllow.textContent = 'Allow for this run';
      runAllow.title =
        'Stop asking about this command for the rest of this run. Not remembered: the ' +
        'next run asks again.';
      runAllow.onclick = () => answer(true, undefined, false, true);
      row.append(runAllow);
    }

    // "Always allow" is offered only where it would actually do something -- and whether it
    // would is COMPUTED BY THE GATE and sent as can_remember, not guessed here.
    //
    // This used to test "irreversible" alone, which is a much weaker condition than the one
    // the matcher applies. A command carrying a ";" or a pipe can never be matched by a
    // stored prefix (allowlisting "pytest" must not authorise "pytest; rm -rf ~"), nor can
    // one the classifier could not fully parse. Both were offered the button anyway, and
    // both wrote a rule that sat in settings doing nothing: MEASURED, 19 of them in one
    // workspace against a run that answered 35 cards. A button that says "you will not be
    // asked again" and leaves you being asked again is worse than no button at all.
    if (payload.command && !payload.irreversible && payload.can_remember) {
      const always = document.createElement('button');
      always.className = 'ghost';
      always.textContent = 'Always allow';
      always.title = 'Remember this exact command, for this run and every later one: ' +
        payload.command;
      always.onclick = () => answer(true, payload.command);
      row.append(always);
    }
    card.append(h, caps, pre, row);
    if (payload.irreversible) {
      const note = document.createElement('div');
      note.className = 'warnbox';
      // Irreversibility is a property above both the blanket switch and the persistent
      // prefix allowlist. "Always allow" is not offered on irreversible command cards
      // (see above); opaque script consent for the rest of a run is digest-bound and
      // separate. The note must match the gate, not a prior experiment that let named
      // allowlists beat irreversibility.
      note.textContent = payload.command
        ? 'This cannot be undone. Neither auto-approve-exec nor a remembered command ' +
          'on the allowlist can skip this card -- approve it each time it appears.'
        : 'This overwrites existing content, so auto-approve-writes will not skip it. ' +
          '"Allow writes for this run" covers the rest of this run only.';
      card.append(note);
    }
    add(card, '');
    busy(true, 'Waiting for you', 'WAITING');
  }

  if (kind === 'perf') {
    const s = payload.sample;
    // The context figure moved OUT of this line and into the meter above it. Leaving a
    // second copy here would be two readings of the same number that can disagree by a
    // frame, and the throughput line is about speed.
    $('perf').textContent =
      'ttft ' + Math.round(s.ttft_ms) + 'ms · prefill ' + s.prefill_tok_per_s.toFixed(0) +
      ' tok/s · decode ' + s.decode_tok_per_s.toFixed(1) + ' tok/s';
    paintContext(s.context_used, s.context_max, s.compactions);
  }

  // The plan handoff. Rendered through the SAME markdown pipeline as an answer, because
  // it is the same kind of thing to read and a plan shown as preformatted text is a plan
  // nobody reads. Not routed through the HITL approver: that gate answers "may this call
  // run", and this is not a call -- it is a decision about what to do next.
  if (kind === 'plan_ready') {
    closeBubble();
    const d = document.createElement('div');
    d.className = 'card';
    const h = document.createElement('h4');
    h.textContent = 'Plan ready';
    const body = document.createElement('div');
    body.className = 'assistant';
    d.append(h, body);
    const row = document.createElement('div');
    row.className = 'row';
    const go = document.createElement('button');
    go.className = 'primary';
    go.textContent = 'Start implementing';
    const stay = document.createElement('button');
    stay.className = 'ghost';
    stay.textContent = 'Keep planning';
    const decide = (start) => {
      go.disabled = true;
      stay.disabled = true;
      // No payload. The extension host is holding the plan text the sidecar sent it, and
      // a mission is not a thing to accept back from a view.
      api.postMessage({ kind: start ? 'start_implementing' : 'keep_planning' });
    };
    go.onclick = () => decide(true);
    stay.onclick = () => decide(false);
    row.append(go, stay);
    d.append(row);
    add(d, '');
    // Fed whole rather than streamed: it arrived whole, and a plan that types itself out
    // while its own buttons sit underneath it is theatre.
    const ctx = newMdCtx(body);
    renderMd(ctx, ctx.stream.feed(payload.plan));
    renderMd(ctx, ctx.stream.finish());
    flushQueue();
  }

  if (kind === 'history') paintHistory(payload.runs);

  if (kind === 'resumed') {
    // The pane was showing a different conversation, so it is cleared rather than appended
    // to: painting the old feed under a resumed session would show one conversation while
    // the model answers from another.
    //
    // The .msg children ONLY. feed.textContent = '' takes the live row with them -- and
    // with it the orb's WebGL context, which does not come back. Same rule as run_start.
    closeBubble();
    for (const el of [...feed.querySelectorAll('.msg')]) el.remove();
    $('plan').textContent = '';
    $('mission').textContent = payload.mission || '';

    if (payload.omitted > 0) {
      const o = document.createElement('div');
      o.className = 'ended';
      o.textContent = payload.omitted + ' earlier turn(s) are not shown here — the ' +
                      'transcript was capped for the pane. The model still has all of them.';
      add(o, '');
    }
    for (const e of payload.transcript || []) {
      if (e.role === 'user') {
        const d = document.createElement('div');
        const who = document.createElement('div');
        who.className = 'who';
        who.textContent = 'You';
        const body = document.createElement('div');
        body.textContent = e.text;
        d.append(who, body);
        add(d, 'user');
      } else if (e.role === 'assistant') {
        // Plain text, not streamed markdown: renderMd drives a streaming parser whose
        // state belongs to the LIVE turn, and feeding a replay through it would leave that
        // state pointing at a conversation that already ended.
        const d = document.createElement('div');
        d.textContent = e.text;
        add(d, 'assistant');
      } else {
        // The same collapsible row a live tool turn gets, so a resumed conversation reads
        // identically to the one it is a picture of.
        const d = document.createElement('details');
        const sum = document.createElement('summary');
        const dot = document.createElement('span');
        dot.className = 'dot ' + (e.is_error ? 'failed' : 'ok');
        const n = document.createElement('span');
        n.className = 'name';
        n.textContent = e.tool;
        const a = document.createElement('span');
        a.className = 'args';
        a.textContent = e.args || '';
        sum.append(dot, n, a);
        const pre = document.createElement('pre');
        pre.textContent = e.truncated ? e.text + '\\n\\n… truncated for display' : e.text;
        d.append(sum, pre);
        add(d, '');
      }
    }
    const b = document.createElement('div');
    b.className = 'ended';
    b.textContent = 'Reopened · ' + payload.turns + ' turns restored. Send a message to carry on.';
    add(b, '');
  }

  if (kind === 'run_end') {
    closeBubble();
    // A CONVERSATION THAT HANDED BACK IS NOT A RUN THAT STOPPED.
    //
    // 'completed' is evidential and a planning session can never earn it -- it declares no
    // contract and writes no code -- so both of these would otherwise print "Stopped",
    // which is the same word this UI uses for a run that ran out of budget mid-edit. The
    // two endings are the mode working, and they say so.
    const asked = payload.termination_reason === 'awaiting_user' &&
                  (lastTurnTool === 'ask_user' || lastTurnTool === 'ask_question');
    const paused = payload.termination_reason === 'awaiting_user' && !asked;
    const yielded = asked || paused || payload.termination_reason === 'plan_ready';
    const d = document.createElement('div');
    d.className = 'ended';
    let t = 'Ended: ' + payload.termination_reason + ' · ' + payload.iterations + ' turn(s)';
    let label = payload.completed ? 'Complete' : 'Stopped';
    if (payload.termination_reason === 'plan_ready') {
      t = 'the plan is above — approve it, or keep talking to change it';
      label = 'Your turn';
    } else if (asked) {
      // A question halt. The card is the reply surface when we drew one; the composer
      // is the fallback when we could not find a choice set. Both are "your turn".
      t = lastQuestionCardDrawn
        ? 'pick an option above and the conversation continues'
        : 'type your answer in the box below and the conversation continues';
      label = 'Your turn';
    } else if (paused) {
      // Plan mode yields after inert turns with the SAME reason as ask_user
      // (\'awaiting_user\'). That is not a question. Measured: r-18cf17272d7adbb0-223bccea
      // ended this way after list_dir of an asset pack, and the old copy made it look
      // like a question box that never appeared.
      t = 'the run paused without a question · send a message to continue';
      label = 'Paused';
    }
    // The checklist is the model's own progress display and holds no authority over the
    // ending -- but "done, with items still open on its own list" is worth a human's
    // glance, so the disagreement is printed rather than resolved silently.
    if (payload.completed && payload.unfinished_items > 0) {
      t += ' · ' + payload.unfinished_items + ' item(s) still open on its own checklist';
    }
    d.innerHTML = '<b>' + label + '</b> — ';
    d.append(document.createTextNode(t));
    // Through add(), like every other block. Appending directly is what let the footer
    // land while the last paragraph of the answer was still being typed above it.
    add(d, '');
    // The run is over: nothing more is coming, so there is nothing left for the typewriter
    // to smooth. Holding text back now is not pacing, it is a transcript that disagrees
    // with the "Ended" line sitting under it.
    flushQueue();
    // Green only for an ending that is actually complete. "Stopped" covers the wall clock,
    // the iteration cap and a cancel, and none of those are a success (S14). A yield is
    // neither: nothing failed and nothing finished, the run is waiting on a person, and
    // WAITING is the hue the orb already has for exactly that.
    finish(label, yielded ? 'WAITING' : payload.completed ? 'DONE' : 'FAILED');
    // The button is hidden by the busy class again, but a disabled control that comes
    // back disabled is how the next run ends up unstoppable.
    $('stop').disabled = false;
    // The history panel is stale the moment a run ends, and it is the moment someone is
    // most likely to look at it.
    if ($('history').classList.contains('open')) api.postMessage({ kind: 'history' });
  }

  if (kind === 'idle') goIdle();
});

// LAST, after every listener above is attached. The host replays its state -- model
// status, settings -- in answer to this, so a view that was rebuilt (window reload; the
// hidden-panel destroy this used to suffer from) starts from the truth instead of from
// blank. The host posting on its own schedule was a race: a message sent before this
// script ran was silently dropped, which is how a loaded model came back "unloaded".
api.postMessage({ kind: 'ready' });
`;
}

// WHICH VIEW CODE IS ACTUALLY RUNNING, shown in the settings drawer.
//
// The sidecar binary is re-read from disk on every run; this script is read once, when the
// window loads. So an editor left open across a reinstall runs the NEW sidecar against the
// OLD renderer -- and the event log cannot see the difference, because the log is written
// by the sidecar. That combination has already cost one round of "the C++ fix landed and
// the view fix did not", with no way to tell it from an incomplete fix.
//
// Derived from the shipped script text itself rather than from a version field or a build
// step, so it cannot be stale by construction: it changes exactly when the view changes,
// and `git stash`-ing a fix changes it back.
function viewBuildId(text: string): string {
  // FNV-1a, 32-bit. A short label a human reads off the screen and compares, not a
  // cryptographic digest -- collisions between two builds someone is choosing between are
  // not the failure mode here.
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export function webviewHtml(nonce: string): string {
  const body = script();
  const build = viewBuildId(body);
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>${styles()}</style></head><body>${markup()}
<script nonce="${nonce}">const VIEW_BUILD = ${JSON.stringify(build)};
${body}</script></body></html>`;
}
