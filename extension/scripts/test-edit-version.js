#!/usr/bin/env node
// Focused contract checks for optimistic-concurrency helpers (no VS Code host).
"use strict";

const assert = require("assert");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const src = path.join(root, "src/edit_version.ts");
const out = path.join(root, "out/edit_version.js");

// Ensure the compiled helper exists (tsc output).
if (!fs.existsSync(out)) {
  execFileSync("npx", ["tsc", "-p", "."], { cwd: root, stdio: "inherit" });
}

const { contentVersion, editPreconditionError } = require(out);

const sample = "hello\n";
const hex = contentVersion(sample);
assert.strictEqual(hex.length, 64);
assert.strictEqual(
  hex,
  crypto.createHash("sha256").update(sample, "utf8").digest("hex")
);

assert.strictEqual(editPreconditionError(undefined, "", true), undefined);
assert.match(
  editPreconditionError(sample, "", true) || "",
  /expected_absent/
);
assert.match(
  editPreconditionError(undefined, hex, false) || "",
  /absent/
);
assert.strictEqual(editPreconditionError(sample, hex, false), undefined);
assert.match(
  editPreconditionError("dirty\n", hex, false) || "",
  /version conflict/
);

// Protocol surface carries the fields the extension compares.
const proto = fs.readFileSync(path.join(root, "src/protocol.generated.ts"), "utf8");
assert.match(proto, /expected_version:\s*string/);
assert.match(proto, /expected_absent:\s*boolean/);

console.log("test-edit-version: ok");
void src;
