// Render the real activity orb in a plain browser, big enough to judge.
//
// The orb is 34px in the sidebar, which is the right size to LIVE with and the wrong size
// to develop against: a dispersion fringe and a key highlight are one pixel each at that
// scale, so a change that ruins them is invisible until it ships. This page runs the same
// orb.ts source at whatever size is asked for, over the real sidebar background, with the
// state settable from the console.
//
//   node scripts/preview-orb.js > /tmp/orb.html            # after `npm run compile`
//   node scripts/preview-orb.js 200 light > /tmp/orb.html  # size in px, and the backdrop
//
// Then drive it:  window.__orb.state('TOOL')  /  window.__orb.impulse('tool')
//
// Not part of the extension.

const path = require("path");
const { webviewHtml } = require(path.join(__dirname, "..", "out", "webview.js"));

const NONCE = "preview";
const SIZE = Number(process.argv[2] || 200);
const THEME = process.argv[3] === "light" ? "light" : "dark";

const PALETTE =
  THEME === "light"
    ? { fg: "#1c1b19", side: "#f3f3f3", editor: "#ffffff", link: "#0a66c2" }
    : { fg: "#e6e6e6", side: "#1e1e1e", editor: "#1e1e1e", link: "#4daafc" };

// Everything the editor would normally supply, plus the host shim.
const SHIM = `
<style nonce="${NONCE}">
  :root {
    --vscode-foreground: ${PALETTE.fg};
    --vscode-sideBar-background: ${PALETTE.side};
    --vscode-editor-background: ${PALETTE.editor};
    --vscode-input-background: ${PALETTE.editor};
    --vscode-input-foreground: ${PALETTE.fg};
    --vscode-editor-font-family: ui-monospace, Menlo, monospace;
    --vscode-textLink-foreground: ${PALETTE.link};
    /* The point of the page: the shipped size is 34px, this is not. */
    --orb-size: ${SIZE}px;
  }
  body { background: ${PALETTE.side}; }
  /* Room for a big bead, and a strip of every state's label underneath it. */
  #headRow { align-items: center; }
  #orb { opacity: 1 !important; }
  #states {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 9;
    display: flex; gap: 6px; padding: 10px; justify-content: center;
    background: ${PALETTE.side}; border-top: 1px solid rgba(128,128,128,.3);
  }
  #states button { background: rgba(128,128,128,.18); color: ${PALETTE.fg}; }
</style>
<script nonce="${NONCE}">
  window.acquireVsCodeApi = () => ({ postMessage() {}, getState() {}, setState() {} });
  // A background tab suspends requestAnimationFrame, which would freeze the orb and read
  // as a rendering bug rather than as a suspended page. The shipped webview is visible
  // inside the editor and uses the real rAF, so this stays in the harness.
  if (document.hidden) window.requestAnimationFrame = (fn) => setTimeout(fn, 8);
</script>`;

const DRIVER = `
<div id="states"></div>
<script nonce="${NONCE}">
  const NAMES = ['IDLE','THINKING','WRITING','TOOL','WAITING','DONE','FAILED'];
  const host = document.getElementById('states');
  for (const n of NAMES) {
    const b = document.createElement('button');
    b.textContent = n;
    b.onclick = () => { window.__orb.state(n); document.getElementById('statusText').textContent = n; };
    host.append(b);
  }
  const t = document.createElement('button');
  t.textContent = 'impulse';
  t.onclick = () => window.__orb.impulse('tool');
  host.append(t);

  document.getElementById('mission').textContent = 'Orb preview — ${SIZE}px, ${THEME}';
  document.body.classList.add('busy');

  // ?state=TOOL lets a headless capture pick which one it is looking at, so the seven
  // states can be screenshotted and compared without anyone clicking anything.
  const q = new URLSearchParams(location.search);
  const want = q.get('state');
  if (want && NAMES.includes(want)) {
    window.__orb.state(want);
    document.getElementById('statusText').textContent = want;
  }

  // ?snap=<phase> freezes the orb: springs on their targets, motion phases pinned. A
  // headless screenshotter otherwise races the animation -- the same command produced a
  // settled bead one run and a half-transitioned one the next, and two shots taken that way
  // cannot be compared.
  if (q.has('snap')) {
    window.__orb.snap(Number(q.get('snap')) || 6);
    // Read on 'load', not now: the orb mounts on DOMContentLoaded, and which path it took
    // -- WebGL2 or the CSS bead -- is not decided until it does.
    window.addEventListener('load', () => {
      const r = window.__orb.read();
      document.getElementById('mission').textContent =
        r.state + '  L' + r.colour.L.toFixed(2) + ' C' + r.colour.C.toFixed(2) +
        ' H' + r.colour.H.toFixed(0) + '  e' + r.energy.toFixed(2) +
        (r.fallback ? '  [CSS fallback]' : '  [webgl2]');
    });
  }
</script>`;

let html = webviewHtml(NONCE);
html = html.replace("<body>", "<body>" + SHIM);
html = html.replace("</body>", DRIVER + "</body>");
process.stdout.write(html);
