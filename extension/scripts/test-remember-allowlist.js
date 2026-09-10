#!/usr/bin/env node
// remember() ingest + allowed_commands serialization, no VS Code host.
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const out = path.join(root, "out/allowlist.js");
if (!fs.existsSync(out)) {
  execFileSync("npx", ["tsc", "-p", "."], { cwd: root, stdio: "inherit" });
}

const { rememberableCommand, serializeAllowedCommands } = require(out);

// --- remember / ingest -------------------------------------------------------

assert.strictEqual(rememberableCommand("pytest"), "pytest");
assert.strictEqual(rememberableCommand("  npm test  "), "npm test");
assert.strictEqual(rememberableCommand(""), undefined);
assert.strictEqual(rememberableCommand("   "), undefined);

assert.strictEqual(
  rememberableCommand("pytest\nrm -rf /"),
  undefined,
  "LF in the command must not be stored"
);
assert.strictEqual(
  rememberableCommand("pytest\rrm -rf /"),
  undefined,
  "CR in the command must not be stored"
);
assert.strictEqual(
  rememberableCommand("pytest\r\nrm -rf /"),
  undefined,
  "CRLF in the command must not be stored"
);
assert.strictEqual(
  rememberableCommand("  good\nbad  "),
  undefined,
  "trim does not strip interior newlines; still refuse"
);

assert.strictEqual(rememberableCommand("pytest; rm -rf /"), undefined);
assert.strictEqual(rememberableCommand("a | b"), undefined);
assert.strictEqual(rememberableCommand("a && b"), undefined);
assert.strictEqual(rememberableCommand("echo `id`"), undefined);
assert.strictEqual(rememberableCommand("echo $(id)"), undefined);
assert.strictEqual(rememberableCommand("a > b"), undefined);
assert.strictEqual(rememberableCommand("a < b"), undefined);

// --- serialize / wire --------------------------------------------------------

assert.strictEqual(serializeAllowedCommands(["pytest", "npm test"]), "pytest\nnpm test");
assert.strictEqual(serializeAllowedCommands([]), "");
assert.strictEqual(
  serializeAllowedCommands(["pytest", "evil\nrm -rf /", "cargo test"]),
  "pytest\ncargo test",
  "LF-bearing settings entries must be dropped before join"
);
assert.strictEqual(
  serializeAllowedCommands(["ok", "bad\rinjected", "also-ok"]),
  "ok\nalso-ok",
  "CR-bearing settings entries must be dropped before join"
);
assert.strictEqual(
  serializeAllowedCommands(["keep", "poison\r\ncurl evil", "keep2"]),
  "keep\nkeep2"
);
assert.strictEqual(
  serializeAllowedCommands(["", "pytest"]),
  "pytest",
  "empty strings are not allowlist rules"
);
assert.strictEqual(
  serializeAllowedCommands(["pytest", "pytest; rm -rf /", "cargo test", "a | b"]),
  "pytest\ncargo test",
  "shell-chaining settings entries must be dropped before join"
);

// Call sites still use the helpers (not a raw join / unchecked push).
const sidebar = fs.readFileSync(path.join(root, "src/sidebar.ts"), "utf8");
assert.match(sidebar, /rememberableCommand\(/);
assert.match(sidebar, /from ["']\.\/allowlist["']/);

const extension = fs.readFileSync(path.join(root, "src/extension.ts"), "utf8");
assert.match(extension, /serializeAllowedCommands\(/);
assert.match(extension, /from ["']\.\/allowlist["']/);
assert.doesNotMatch(
  extension,
  /allowedCommands["']?\s*,\s*\[\]\)\s*\.join\(\s*["']\\n["']\s*\)/,
  "settingsFromConfig must not raw-join allowedCommands"
);

console.log("test-remember-allowlist: ok");
