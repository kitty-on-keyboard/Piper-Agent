#!/usr/bin/env node
// Test for command allowlist sanitization against newline injection and shell operators.
"use strict";

const assert = require("assert");

const TICK = String.fromCharCode(96);
const SHELL_OPERATOR_REGEX = new RegExp("[;|&" + TICK + "<>]|\\$\\(", "u");

function isValidRememberCommand(cmd) {
  if (typeof cmd !== "string") return false;
  if (/[\r\n]/.test(cmd)) return false;
  const trimmed = cmd.trim();
  if (!trimmed) return false;
  if (SHELL_OPERATOR_REGEX.test(trimmed)) return false;
  return true;
}

// Test filter function used in extension.ts settingsFromConfig()
function sanitizeAllowedCommands(commands) {
  if (!Array.isArray(commands)) return "";
  return commands
    .filter((cmd) => typeof cmd === "string" && !/[\r\n]/.test(cmd))
    .join("\n");
}

// 1. Valid commands should be accepted
assert.strictEqual(isValidRememberCommand("git status"), true);
assert.strictEqual(isValidRememberCommand("pnpm test"), true);
assert.strictEqual(isValidRememberCommand("swift build"), true);

// 2. Commands with newlines or carriage returns must be rejected
assert.strictEqual(isValidRememberCommand("git status\nrm -rf /"), false);
assert.strictEqual(isValidRememberCommand("git status\r\nrm -rf /"), false);
assert.strictEqual(isValidRememberCommand("\nrm -rf /"), false);
assert.strictEqual(isValidRememberCommand("rm -rf /\n"), false);

// 3. Commands with shell operators must be rejected
assert.strictEqual(isValidRememberCommand("git status; rm -rf /"), false);
assert.strictEqual(isValidRememberCommand("git status && rm -rf /"), false);
assert.strictEqual(isValidRememberCommand("git status | grep main"), false);
assert.strictEqual(isValidRememberCommand("echo $(whoami)"), false);
assert.strictEqual(isValidRememberCommand("echo " + TICK + "whoami" + TICK), false);
assert.strictEqual(isValidRememberCommand("cat < /etc/passwd"), false);
assert.strictEqual(isValidRememberCommand("echo test > file"), false);

// 4. Non-string or empty inputs must be rejected
assert.strictEqual(isValidRememberCommand(null), false);
assert.strictEqual(isValidRememberCommand(undefined), false);
assert.strictEqual(isValidRememberCommand(123), false);
assert.strictEqual(isValidRememberCommand("   "), false);

// 5. Filtering test for allowedCommands array
const rawAllowedCommands = [
  "git status",
  "pnpm test",
  "git status\nrm -rf /",
  "touch /tmp/malicious\r\n",
  123,
  null,
  "swift build",
];

const wireString = sanitizeAllowedCommands(rawAllowedCommands);
assert.strictEqual(wireString, "git status\npnpm test\nswift build");

console.log("test-remember-allowlist: ok");
