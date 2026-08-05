// The transcript's ordering invariant, checked against the REAL shipped view source.
//
//   node scripts/verify-transcript-order.js      # after `npm run compile` is NOT needed:
//                                                # this reads src/webview.ts directly
//
// WHAT IT PINS. A block added to the feed cannot reach the screen before text that was
// queued ahead of it, and the queue cannot fall arbitrarily far behind the stream.
//
// MEASURED, against the code this replaced: the tool row landed with 0 of 700 characters
// of the answer on screen, and those 700 characters then took 350 frames -- about six
// seconds at 60fps -- to finish arriving underneath it. That is the whole of the "tool
// calls appear before the text" report, and both halves had their own cause. add()
// appended SYNCHRONOUSLY while text drained from a queue, so a row could overtake text
// outright; and the drain rate was computed from the length of the CURRENT JOB, which
// after the sidecar began streaming token by token was about four characters, so it sat
// on its floor of 2 chars/frame and never caught up.
//
// The block under test is sliced out of webview.ts by its own section markers rather than
// copied, so this cannot pass against a stale duplicate. If the markers move the slice
// fails loudly -- a harness that silently tests zero lines is worse than no harness.

const fs = require("fs");
const path = require("path");

const SRC = fs.readFileSync(path.join(__dirname, "..", "src", "webview.ts"), "utf8");

// Wide enough to include add(), which is the function the reordering actually lived in:
// the drain rate decided HOW FAR behind the text ran, but add() appending synchronously is
// what let a tool row overtake it at all.
const START = "// --- typewriter ---";
const END = "// --- composer ---";
const a = SRC.indexOf(START);
const b = SRC.indexOf(END);
if (a < 0 || b < 0 || b <= a) {
  console.error("FAIL: could not slice the typewriter block; markers moved");
  process.exit(1);
}
const block = SRC.slice(a, b);
if (!/function drain\(/.test(block) || !/function flushQueue\(/.test(block)) {
  console.error("FAIL: sliced block does not contain the functions under test");
  process.exit(1);
}

// --- the DOM stub ---------------------------------------------------------
// Only what the block touches: a node that records the order text and blocks landed in.
const landed = [];
function makeNode(name) {
  return {
    name,
    appendChild(n) { landed.push(n.__text !== undefined ? n.__text : "[" + n.name + "]"); },
    insertBefore(n) { landed.push(n.__text !== undefined ? n.__text : "[" + n.name + "]"); },
  };
}
const feed = makeNode("feed");
feed.scrollHeight = 0;
feed.scrollTop = 0;
feed.clientHeight = 0;
const live = makeNode("live");
const target = makeNode("bubble");
const caret = null;

function makeEl(tag) {
  const el = makeNode(tag);
  el.classList = { add() {}, toggle() {}, remove() {} };
  el.append = () => {};
  el.remove = () => {};
  return el;
}
const document = {
  createTextNode: (t) => ({ __text: t }),
  createElement: (tag) => makeEl(tag),
  getElementById: () => makeEl("stub"),
};
feed.querySelectorAll = () => [];
const window = {};
const $ = () => makeEl("stub");

// Frames are driven by hand so the test is deterministic -- a real rAF would make the
// assertion depend on wall-clock scheduling.
let frame = null;
const requestAnimationFrame = (fn) => { frame = fn; };
function runFrames(n) {
  for (let i = 0; i < n && frame; i++) { const f = frame; frame = null; f(); }
}

const ctx = { target: () => target, afterBlock: false };

const scope = { feed, live, caret, document, requestAnimationFrame, window, $ };
const fn = new Function(
  ...Object.keys(scope),
  block +
    "\nreturn {queue, typeMd, queueOp, add, flushQueue, step, pump, drain, get pending(){return pending;}};"
);
const T = fn(...Object.values(scope));

// --- the case that was broken --------------------------------------------
// 700 characters of answer, streamed as ~4-char tokens in one tick, then a tool row.
const ANSWER = "x".repeat(700);
const tokens = [];
for (let i = 0; i < ANSWER.length; i += 4) tokens.push(ANSWER.slice(i, i + 4));
for (const t of tokens) T.typeMd(ctx, t, false);
// Through add(), exactly as the `turn` handler does -- this is the call that used to hit
// the feed synchronously while the text above it was still trickling out.
const row = makeEl("details");
row.name = "TOOL ROW";
T.add(row, "");

// Drain frame by frame and record when the tool row appears relative to the text.
let frames = 0;
while (T.queue.length && frames < 5000) { runFrames(1); frames++; }

const rowAt = landed.indexOf("[TOOL ROW]");
const charsBefore = landed.slice(0, rowAt).reduce((n, s) => n + (s[0] === "[" ? 0 : s.length), 0);

let failed = false;
const check = (ok, label, detail) => {
  console.log((ok ? "PASS " : "FAIL ") + label + (detail ? " -- " + detail : ""));
  if (!ok) failed = true;
};

check(rowAt >= 0, "the tool row reached the screen");
check(
  charsBefore === ANSWER.length,
  "the tool row lands only after every character before it",
  charsBefore + " of " + ANSWER.length + " characters were on screen first"
);
check(frames < 60, "the backlog clears within a second", frames + " frames for 700 chars");
check(T.pending === 0, "the pending counter returns to zero", "pending=" + T.pending);

// flushQueue empties everything at once.
landed.length = 0;
for (const t of tokens) T.typeMd(ctx, t, false);
const foot = makeEl("div"); foot.name = "FOOTER"; T.add(foot, "");
T.flushQueue();
check(T.queue.length === 0, "flushQueue empties the queue");
check(landed[landed.length - 1] === "[FOOTER]", "flushQueue keeps the footer last");

process.exit(failed ? 1 : 0);
