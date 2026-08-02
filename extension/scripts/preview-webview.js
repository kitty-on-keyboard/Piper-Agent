// Render the real sidebar view script in a plain browser, with the sidecar faked.
//
// This exists because the markdown work is a RENDERING change, and the trace test in
// verify-markdown-stream.js only proves the parser matches the C++ -- it says nothing about
// what the DOM ends up looking like. This drives the actual `webviewHtml()` output, streams
// a sample answer into it token by token the way the sidecar does, and leaves a page that
// can be screenshotted.
//
//   node scripts/preview-webview.js > /tmp/preview.html      # after `npm run compile`
//
// Not part of the extension. VS Code's own theme variables are absent outside the editor,
// so a small fallback palette is injected to keep the page legible.

const path = require("path");
const { webviewHtml } = require(path.join(__dirname, "..", "out", "webview.js"));

const NONCE = "preview";

const SAMPLE = `Here is what I found.

## The cause

The bug is in \`parse_header\` — it reads past the end of the buffer. Two things go wrong:

- the length check uses \`<=\` instead of \`<\`
- the fallback path never runs
  - because \`ok\` is set before the check
    - which also hides the real error

## The fix

\`\`\`cpp
int parse_header(std::span<const std::byte> in) {
    if (in.size() < kHeaderBytes) return -1;   // was <=
    return decode(in.first(kHeaderBytes));
}
\`\`\`

That closes it. Run \`ctest --preset gate\` to confirm.

1. apply the patch
2. rebuild
3. re-run the gate
`;

// The theme variables the editor would normally supply, plus the host shim.
const SHIM = `
<style nonce="${NONCE}">
  :root {
    --vscode-foreground: #e6e6e6;
    --vscode-sideBar-background: #1e1e1e;
    --vscode-textLink-foreground: #4daafc;
  }
  body { background: #1e1e1e; }
</style>
<script nonce="${NONCE}">
  window.acquireVsCodeApi = () => ({ postMessage() {}, getState() {}, setState() {} });

  // The typewriter drains on requestAnimationFrame, which browsers SUSPEND for a page whose
  // document.hidden is true -- as it is in an automated/background tab. Without this the
  // preview freezes part-way through and looks like a rendering bug. The shipped webview is
  // visible inside the editor and uses the real rAF; this shim is the harness's problem
  // only, so it must never move into webview.ts.
  if (document.hidden) window.requestAnimationFrame = (fn) => setTimeout(fn, 8);
</script>`;

// Streams the sample in small chunks, as `lmp/token` does -- ~51 notifications per turn on
// the real sidecar -- so the preview exercises the incremental path, not a single slab.
const DRIVER = `
<script nonce="${NONCE}">
  const post = (kind, payload) => window.dispatchEvent(
    new MessageEvent('message', { data: { kind, payload } }));

  post('run_start', { mission: 'Fix the header parser' });
  const text = ${JSON.stringify(SAMPLE)};

  // Deterministic chunk sizes: a preview that renders differently every run is no good for
  // comparing screenshots.
  let seed = 12345;
  const nextChunk = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return 3 + (seed % 6); };

  let i = 0;
  const sendOne = () => {
    const n = nextChunk();
    post('token', { channel: 'answer', text: text.slice(i, i + n) });
    i += n;
  };

  if (document.hidden) {
    // setTimeout is clamped to ~1s in a background tab, which would make this take minutes.
    // Post every chunk synchronously instead: the parser still sees the text as ~80 separate
    // feeds, so the incremental path is exercised -- only the wall-clock pacing is dropped.
    while (i < text.length) sendOne();
    post('run_end', { completed: true });
  } else {
    (function tick() {
      if (i >= text.length) { post('run_end', { completed: true }); return; }
      sendOne();
      setTimeout(tick, 4);
    })();
  }
</script>`;

let html = webviewHtml(NONCE);
html = html.replace("<body>", "<body>" + SHIM);
html = html.replace("</body>", DRIVER + "</body>");
process.stdout.write(html);
