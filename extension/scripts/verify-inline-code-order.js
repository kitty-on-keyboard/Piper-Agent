// Inline code spans must render WHERE THEY WERE WRITTEN, not collected at the end.
//
//   node scripts/verify-inline-code-order.js     # reads src/webview.ts directly
//
// WHAT IT PINS. The renderer has two writers into the same container and they must agree
// on where the end of the written content is. `drain` puts text in with
// `insertBefore(node, caret)` when the caret is a child of the target; `applyMd` used to
// put ELEMENTS in with a plain `appendChild`. Whenever the caret sat in that container and
// was not the last child, the two disagreed permanently: prose accumulated before the
// caret and every <code>, <pre>, <h*> and <ul> piled up after it, in the order created.
//
// MEASURED, from a real run's transcript. The model wrote:
//
//   ...identified it as a Swift package (`ResMon.xcodeproj`, `Package.swift`). The
//   directory `Sources/ResMon/` seems to be...
//
// and the bubble rendered "a Swift package (, )... The directory  seems to be..." with
// every code span swept to the end and run together, because adjacent <code> siblings with
// no text between them read as one span. The parser was never involved -- it emits
// text/inlineOpen/text/inlineClose in the right order at every chunk size, which
// verify-markdown-stream.js already proves against the C++ original.
//
// The trigger is `applyMd`'s caret re-anchoring being conditional (`ctx === mdCtx`): a
// context still draining after a new bubble has opened stops re-anchoring, and from that
// point on the two writers diverge. Rather than widen the condition -- which leaves the
// same latent disagreement one refactor away -- both writers now use one rule. This drives
// the renderer in exactly that state and checks the text comes out in source order.
//
// Sliced out of webview.ts by its own section markers rather than copied, for the same
// reason verify-transcript-order.js is: a harness that tests a stale duplicate is worse
// than no harness.

const fs = require("fs");
const path = require("path");

const SRC = fs.readFileSync(path.join(__dirname, "..", "src", "webview.ts"), "utf8");

const START = "// --- typewriter ---";
const END = "// --- composer ---";
const a = SRC.indexOf(START);
const b = SRC.indexOf(END);
if (a < 0 || b < 0 || b <= a) {
  console.error("FAIL: could not slice the typewriter block; markers moved");
  process.exit(1);
}
const block = SRC.slice(a, b);
if (!/function applyMd\(/.test(block) || !/function newMdCtx\(/.test(block)) {
  console.error("FAIL: sliced block does not contain the functions under test");
  process.exit(1);
}

// --- an ordering DOM stub -------------------------------------------------
// Unlike verify-transcript-order.js's stub, this one keeps real parent/child order:
// the defect IS an ordering defect, so a stub that only records arrival order cannot see
// it -- every node does arrive, in the order it was created.
function makeNode(name) {
  const n = {
    name,
    children: [],
    parentNode: null,
    detach_(c) {
      const i = n.children.indexOf(c);
      if (i >= 0) n.children.splice(i, 1);
    },
    appendChild(c) {
      if (c.parentNode) c.parentNode.detach_(c);
      c.parentNode = n;
      n.children.push(c);
      return c;
    },
    insertBefore(c, ref) {
      if (c.parentNode) c.parentNode.detach_(c);
      c.parentNode = n;
      const i = n.children.indexOf(ref);
      n.children.splice(i < 0 ? n.children.length : i, 0, c);
      return c;
    },
    setAttribute() {},
    remove() {},
  };
  n.classList = { add() {}, toggle() {}, remove() {} };
  n.append = (...cs) => cs.forEach((c) => n.appendChild(c));
  return n;
}

// Walk in document order. Inline code is wrapped in backticks so a span that moved is
// visible in the diff rather than merely mis-parented.
const TICK = String.fromCharCode(96);
function textOf(node) {
  if (node.__text !== undefined) return node.__text;
  const inner = node.children.map(textOf).join("");
  return node.name === "code" ? TICK + inner + TICK : inner;
}

const feed = makeNode("feed");
feed.scrollHeight = 0;
feed.scrollTop = 0;
feed.clientHeight = 0;
feed.querySelectorAll = () => [];
const live = makeNode("live");
const document = {
  createTextNode: (t) => ({ __text: t, parentNode: null, children: [] }),
  createElement: (tag) => makeNode(tag),
  getElementById: () => makeNode("stub"),
};
const window = {};
const $ = () => makeNode("stub");
let frame = null;
const requestAnimationFrame = (fn) => { frame = fn; };

const bubble = makeNode("bubble");
const caret = makeNode("span");
bubble.appendChild(caret);

// THE STATE THAT BREAKS IT, and it is the ordinary one rather than an edge case.
//
// renderMd() QUEUES its work; closeBubble() then sets mdCtx = null SYNCHRONOUSLY the
// moment the turn notification lands. The queue drains over the following animation
// frames, so by the time applyMd runs for an answer of any length, mdCtx is already null
// and its `ctx === mdCtx` caret re-anchoring is false for EVERY op in the turn -- not just
// the tail. That is why a whole paragraph's worth of code spans ended up at the end rather
// than only the last held-back chunk.
//
// So this is `let`, and the sequence below is queue → detach → drain, in that order.
let mdCtx = null;

// newMdCtx builds one per bubble. The parser is not under test here -- its output is the
// hard-coded EVENTS below -- so a stub keeps this harness failing for one reason only.
class MarkdownStream {
  feed() { return []; }
  finish() { return []; }
}

const scope = {
  feed, live, caret, mdCtx, document, requestAnimationFrame, window, $, MarkdownStream,
};
const fn = new Function(
  ...Object.keys(scope),
  block +
    "\nreturn {queue, newMdCtx, renderMd, flushQueue, drain};"
);
const T = fn(...Object.values(scope));

// --- the case that was broken --------------------------------------------
// The parser's own output for the measured sentence, at any chunk size. Hard-coded rather
// than re-derived so this harness tests the RENDERER and fails for one reason only;
// verify-markdown-stream.js is what pins the parser that produces it.
const EVENTS = [
  { kind: "text", text: "identified it as a Swift package (" },
  { kind: "inlineOpen" },
  { kind: "text", text: "ResMon.xcodeproj" },
  { kind: "inlineClose" },
  { kind: "text", text: ", " },
  { kind: "inlineOpen" },
  { kind: "text", text: "Package.swift" },
  { kind: "inlineClose" },
  { kind: "text", text: "). The directory " },
  { kind: "inlineOpen" },
  { kind: "text", text: "Sources/ResMon/" },
  { kind: "inlineClose" },
  { kind: "text", text: " seems to be the main source folder." },
];

const EXPECT =
  "identified it as a Swift package (" + TICK + "ResMon.xcodeproj" + TICK + ", " +
  TICK + "Package.swift" + TICK + "). The directory " +
  TICK + "Sources/ResMon/" + TICK + " seems to be the main source folder.";

// The real sequence: the answer is queued while the context is current, the turn
// notification detaches it, and only then does the typewriter drain.
const ctx = T.newMdCtx(bubble);
mdCtx = ctx;
T.renderMd(ctx, EVENTS);
mdCtx = null;      // what closeBubble() does, synchronously, with the queue still full
T.flushQueue();

const got = textOf(bubble);

let failed = false;
const check = (ok, label, detail) => {
  console.log((ok ? "PASS " : "FAIL ") + label + (detail ? "\n  -- " + detail : ""));
  if (!ok) failed = true;
};

check(
  got === EXPECT,
  "inline code renders in source order with a caret parked in the bubble",
  got === EXPECT ? "" : "expected:\n     " + EXPECT + "\n  got:\n     " + got
);

// The specific corruption, called out on its own so a regression names itself rather than
// showing up as a long string diff: prose with the code spans cut out of it.
check(
  !/\(, \)/.test(got),
  "code spans are not swept out of the sentence they were written in"
);

// And the caret is still somewhere in the bubble afterwards -- the fix must not achieve
// its ordering by dropping the write cursor on the floor, which would stop the next
// token landing in the right place.
check(caret.parentNode !== null, "the caret survives rendering");

process.exit(failed ? 1 : 0);
