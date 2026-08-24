// Parses the webview's inline script, which tsc CANNOT check.
//
// webview.ts builds the whole view as a TEMPLATE LITERAL, so to the compiler it is one
// string: `tsc -p .` passes on a script that cannot parse in a browser. That is not a
// hypothetical -- an escape written `\n` instead of `\\n` is resolved by TypeScript,
// reaches the browser as a REAL newline inside a string literal, and kills the entire view
// at parse time. It has shipped that way before, and it happened again while the resume
// transcript was being written. This is the check that caught it.
const vm = require("vm");
const { webviewHtml } = require("../out/webview.js");

const html = webviewHtml("test-nonce");
const blocks = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (blocks.length === 0) {
  console.error("FAIL: no <script> block in the view HTML");
  process.exit(1);
}

let failed = 0;
blocks.forEach((src, i) => {
  try {
    new vm.Script(src, { filename: `view-script-${i}.js` });
  } catch (e) {
    failed++;
    console.error(`FAIL: view script block ${i} does not parse: ${e.message}`);
  }
});

// The other rule this file exists to hold: the feed's live row is permanent, and clearing
// the feed with textContent takes it -- and the orb's WebGL context, which does not come
// back. Every wipe must go through the .msg children.
const all = blocks.join("\n");
// Code lines only. The rule is DESCRIBED in comments at both places that clear the feed
// correctly, so a naive substring search matches the documentation and fails on a file
// that is right -- which is a guard nobody keeps.
const codeLines = all
  .split("\n")
  .filter((l) => !l.trim().startsWith("//"))
  .join("\n");
if (/(^|[^.\w])feed\.textContent\s*=|\$\('feed'\)\.textContent\s*=/.test(codeLines)) {
  failed++;
  console.error("FAIL: the feed is cleared with textContent, which destroys #live and the");
  console.error("      orb's WebGL context. Remove the .msg children instead.");
}

if (failed > 0) process.exit(1);
console.log(`ok: ${blocks.length} view script block(s) parse, ${all.length} chars`);
