// The sidebar's markup, styles and view script (spec S12.2).
//
// Split out of sidebar.ts, which owns the protocol wiring: this file is the surface the
// user actually looks at, and mixing a stylesheet into the class that dispatches
// notifications made both harder to read.
//
// THREE THINGS THIS FILE IS CAREFUL ABOUT
//
// 1. The model does not stream. `Observer::on_token` fires ONCE per turn with the whole
//    reasoning block and once with the whole answer, so text arrives in slabs after a
//    silence as long as the turn took. The typewriter is therefore not decoration over a
//    stream -- it IS the stream, reconstructed at the view layer, and the thinking
//    animation covers the gap where genuinely nothing is arriving.
//
// 2. Everything is themed from VS Code variables, so it follows the editor into light,
//    dark and high contrast rather than looking correct in whichever one it was built in.
//
// 3. The CSP allows inline styles and exactly one nonced script. No network, no CDN
//    fonts, no remote anything -- the sidecar is local and the view has no business
//    reaching off the machine.

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

/* --- the thinking indicator --------------------------------------------- */
/* A conic ring that rotates while the model is generating, with a soft bloom behind
   it. Present only while there is genuinely something happening -- an idle spinner
   teaches people to ignore it. */
#orb {
  width: 14px; height: 14px; border-radius: 50%; flex: none;
  background: conic-gradient(from 0deg, transparent 0deg, var(--accent) 120deg,
              color-mix(in srgb, var(--accent) 40%, transparent) 240deg, transparent 360deg);
  -webkit-mask: radial-gradient(circle, transparent 54%, #000 56%);
  mask: radial-gradient(circle, transparent 54%, #000 56%);
  animation: spin 1.1s linear infinite;
  opacity: 0; transition: opacity .25s var(--ease);
}
#orb::after {
  content: ""; position: absolute; width: 14px; height: 14px; border-radius: 50%;
  background: var(--accent); filter: blur(7px); opacity: .35;
  animation: breathe 2.2s ease-in-out infinite;
}
body.busy #orb { opacity: 1; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes breathe { 0%,100% { opacity:.18; transform:scale(.85);} 50% { opacity:.4; transform:scale(1.15);} }

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
  #orb, #orb::after, body.busy #statusText { animation: none; }
  body.busy #statusText { -webkit-text-fill-color: var(--fg); }
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
  <div id="mission"></div>
  <div id="status"><span id="orb"></span><span id="statusText">Idle</span></div>
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

/** The view script. Owns the typewriter queue, the busy state and the DOM. */
function script(): string {
  return `
const api = acquireVsCodeApi();
const $ = (id) => document.getElementById(id);
const feed = $('feed');

let inFlight = false;
let bubble = null;          // the assistant bubble currently being typed into
let caret = null;

// --- typewriter -----------------------------------------------------------
// The sidecar hands over a whole turn's text at once, so this is what turns a slab
// into something readable as it lands. The drain rate scales with how far behind it
// is: a short answer types at a human pace, a 4,000-character one does not take a
// minute to appear.
const queue = [];
let typing = false;

function typeInto(node, text) {
  queue.push({ node, text, at: 0 });
  if (!typing) { typing = true; requestAnimationFrame(step); }
}

function step() {
  const job = queue[0];
  if (!job) { typing = false; return; }
  const left = job.text.length - job.at;
  const rate = Math.max(2, Math.ceil(left / 28));
  const chunk = job.text.slice(job.at, job.at + rate);
  job.at += rate;
  job.node.insertBefore(document.createTextNode(chunk), job.node.querySelector('.caret'));
  const pinned = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 60;
  if (pinned) feed.scrollTop = feed.scrollHeight;
  if (job.at >= job.text.length) queue.shift();
  requestAnimationFrame(step);
}

// --- state ----------------------------------------------------------------
function busy(on, label) {
  inFlight = on;
  document.body.classList.toggle('busy', on);
  $('statusText').textContent = label;
  $('hint').textContent = on
    ? 'Your message reaches the run at the next turn boundary'
    : 'Your message continues the conversation';
}

function add(el, cls) {
  el.className = 'msg ' + (cls || '');
  feed.append(el);
  feed.scrollTop = feed.scrollHeight;
  return el;
}

function closeBubble() {
  if (caret) { caret.remove(); caret = null; }
  bubble = null;
}

function openBubble() {
  closeBubble();
  bubble = add(document.createElement('div'), 'assistant');
  caret = document.createElement('span');
  caret.className = 'caret';
  bubble.append(caret);
  return bubble;
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
busy(false, 'Idle');

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
    busy(true, 'Thinking');
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
    busy(true, payload.steering ? 'Steering at the next turn' : 'Thinking');
  }

  if (kind === 'token') {
    if (payload.channel === 'thinking') {
      const d = document.createElement('details');
      d.className = 'thought';
      const s = document.createElement('summary');
      s.textContent = 'Thought for a moment';
      const b = document.createElement('div');
      b.className = 'body';
      b.textContent = payload.text;
      d.append(s, b);
      closeBubble();
      add(d, '');
      busy(true, 'Writing');
    } else {
      typeInto(openBubble(), payload.text);
      busy(true, 'Writing');
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
    busy(true, 'Thinking');
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
    const answer = (ok) => {
      api.postMessage({ kind: 'approve', id: payload.request_id, approved: ok });
      card.remove();
    };
    yes.onclick = () => answer(true);
    no.onclick = () => answer(false);
    row.append(yes, no);
    card.append(h, caps, pre, row);
    add(card, '');
    busy(true, 'Waiting for you');
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
    d.innerHTML = '<b>' + (payload.completed ? 'Complete' : 'Stopped') + '</b> — ';
    d.append(document.createTextNode(t));
    feed.append(d);
    feed.scrollTop = feed.scrollHeight;
  }

  if (kind === 'idle') busy(false, 'Idle');
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
