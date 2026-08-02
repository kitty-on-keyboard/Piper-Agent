// Prove the webview's markdown parser still matches the C++ it was ported from.
//
// The state machine that ships is a STRING inside out/webview.js (the view script is
// injected into the webview, so it cannot be a module). This evaluates that exact string
// rather than a copy -- verifying a copy would verify the copy -- and prints the same
// canonical event trace that bakeoff/markdown_stream/dump.cpp prints for the C++ amalgam.
// The two traces are compared with `diff`; see bakeoff/markdown_stream/README.md.
//
//   node scripts/verify-markdown-stream.js <corpus.txt>   # after `npm run compile`
//
// Corpus lines are backslash-escaped, one document per line, shared with the C++ side.

const fs = require("fs");
const path = require("path");

const { markdownStreamSource } = require(path.join(__dirname, "..", "out", "webview.js"));

// The source declares `class MarkdownStream`; hand it back out of the eval.
const MarkdownStream = eval(markdownStreamSource() + "\nMarkdownStream;");

const unesc = (s) =>
  s.replace(/\\(.)/g, (_, c) => (c === "n" ? "\n" : c === "t" ? "\t" : "\\"));
const esc = (s) =>
  String(s).replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/\t/g, "\\t");

// Adjacent same-kind text runs are merged, matching the C++ dumper: how a chunk is carved
// is a free choice, what it contains is not.
function emit(acc, events) {
  for (const e of events) {
    const last = acc[acc.length - 1];
    if (last && last.kind === e.kind && (e.kind === "text" || e.kind === "codeText")) {
      last.text += e.text;
    } else {
      acc.push({ kind: e.kind, text: e.text || "", info: e.info || "", level: e.level || 0 });
    }
  }
}

const show = (tag, i, acc) => {
  for (const e of acc) {
    process.stdout.write(
      `${tag}[${i}] ${e.kind}|${esc(e.text)}|${esc(e.info)}|${e.level}\n`
    );
  }
};

const corpus = fs.readFileSync(process.argv[2], "utf8").split("\n");
if (corpus[corpus.length - 1] === "") corpus.pop();

corpus.forEach((line, i) => {
  const doc = unesc(line);

  const whole = [];
  const a = new MarkdownStream();
  emit(whole, a.feed(doc));
  emit(whole, a.finish());
  show("whole", i, whole);

  // Byte at a time -- the split-invariance case, and how the sidecar actually streams.
  const bytes = [];
  const b = new MarkdownStream();
  for (const ch of Array.from(doc)) emit(bytes, b.feed(ch));
  emit(bytes, b.finish());
  show("bytes", i, bytes);
});
