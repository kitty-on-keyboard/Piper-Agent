#!/usr/bin/env node
//
// THE VIEW SCRIPT MUST PARSE.
//
// `tsc --noEmit` cannot see this. The webview's script is the body of a TypeScript
// TEMPLATE LITERAL, so the type checker reads the whole thing as one string and is
// perfectly happy with a syntax error inside it -- and the gate was green, the vsix
// packaged, and both editors installed a pane in which NOTHING WORKED.
//
// The failure mode is why this is worth its own check. A syntax error in the view script
// does not break the feature it is in: the script never finishes evaluating, so not one
// listener is attached, and every button in the pane goes dead at once -- send, stop,
// load model, the lot. It presents as "the extension is broken", with nothing in any log
// this repo writes, because the error is in the webview's own console.
//
// MEASURED, 2026-08-15: a regex written as /^image\//. Inside a template literal the
// backslash is consumed when the literal is evaluated, so the webview received
// /^image// -- "SyntaxError: Unexpected token '.'" -- and the whole pane died. It reached
// two installed editors.
//
// Uses the COMPILED out/webview.js so the string is checked exactly as it will be served,
// after the template literal has actually been evaluated. Checking src/webview.ts instead
// would re-introduce the very blindness this exists to remove.

const path = require("node:path");
const vm = require("node:vm");

const here = path.dirname(__dirname);
let webviewHtml;
try {
  ({ webviewHtml } = require(path.join(here, "out", "webview.js")));
} catch (e) {
  console.error("verify-view-script: could not load out/webview.js -- run `npm run compile` first");
  console.error(String(e && e.message ? e.message : e));
  process.exit(1);
}

const html = webviewHtml("noncefortest");
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (scripts.length === 0) {
  console.error("verify-view-script: the page carries no <script> at all");
  process.exit(1);
}

let failed = 0;
scripts.forEach((src, i) => {
  if (src.trim().length === 0) {
    return;
  }
  try {
    // Compiled, never run: this is a syntax check, and the script expects a DOM.
    new vm.Script(src, { filename: `webview-script-${i}.js` });
  } catch (e) {
    failed++;
    console.error(`verify-view-script: script #${i} does not parse`);
    console.error(`  ${e.message}`);
    // The offending line, which is the whole point -- a bare message with no position is
    // the thing that made this expensive to find by hand.
    const m = /webview-script-\d+\.js:(\d+)/.exec(String(e.stack || ""));
    if (m) {
      const n = Number(m[1]);
      const lines = src.split("\n");
      for (let k = Math.max(0, n - 3); k < Math.min(lines.length, n + 2); k++) {
        console.error(`  ${k + 1 === n ? ">" : " "} ${k + 1}: ${lines[k]}`);
      }
    }
  }
});

if (failed > 0) {
  process.exit(1);
}
console.error(`verify-view-script: ok, ${scripts.length} script block(s) parse`);
