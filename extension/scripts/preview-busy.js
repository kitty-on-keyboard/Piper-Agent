// A driven preview of a run IN FLIGHT, for looking at the states the finished-run preview
// cannot show: the stop button (only visible while busy), the context meter under load,
// the compaction chip, and the reasoning disclosure opened.
//
// preview-webview.js drives a run to completion, which is the right default and is exactly
// why it cannot show any of the above -- run_end clears `busy` in the same tick.
//
//   node scripts/preview-busy.js > out/busy.html     # after `npm run compile`

const path = require("path");
const { webviewHtml } = require(path.join(__dirname, "..", "out", "webview.js"));

const NONCE = "preview";

// The theme variables the editor would normally supply. Without these every
// color-mix(... var(--fg) ...) is invalid at computed-value time and silently falls back to
// the property's initial value -- which renders the stop button with a transparent
// background and looks exactly like a missing element rather than a missing variable.
const SHIM =
  '<style nonce="' + NONCE + '">' +
  ":root{--vscode-foreground:#e6e6e6;--vscode-sideBar-background:#1e1e1e;" +
  "--vscode-textLink-foreground:#4daafc;--vscode-input-background:#2a2a2a;" +
  "--vscode-input-foreground:#e6e6e6;}body{background:#1e1e1e}</style>" +
  '<script nonce="' + NONCE + '">' +
  "window.acquireVsCodeApi=()=>({postMessage(){},getState(){},setState(){}});" +
  "</script>";

// Written with escaped newlines because it is embedded in the page as JS source.
const REASONING = [
  "The workspace is empty, so I will start from scratch.",
  "",
  "- define the immutable structs first",
  "- then the actor that owns the Mach calls",
  "",
  "The tricky part is per-core data.",
].join("\\n");

const DRIVER =
  '<script nonce="' + NONCE + '">\n' +
  "(function(){\n" +
  "  var post=function(k,p){window.dispatchEvent(new MessageEvent('message',{data:{kind:k,payload:p}}));};\n" +
  "  setTimeout(function(){\n" +
  "    post('run_start',{mission:'Build a thread-safe key-value store with TTL and transactions',reset:true});\n" +
  "    post('token',{channel:'thinking',text:'" + REASONING + "'});\n" +
  "    post('token',{channel:'answer',text:'Starting with the core store.'});\n" +
  "    post('perf',{sample:{ttft_ms:410,prefill_tok_per_s:1684,decode_tok_per_s:77.4," +
  "context_used:74210,context_max:100096,tokens_generated:240,compactions:2}});\n" +
  "    setTimeout(function(){var d=document.querySelector('details.thought'); if(d){d.open=true;}},400);\n" +
  "  },200);\n" +
  "})();\n" +
  "</script>";

let html = webviewHtml(NONCE);
html = html.replace("<body>", "<body>" + SHIM);
html = html.replace("</body>", DRIVER + "</body>");
process.stdout.write(html);
