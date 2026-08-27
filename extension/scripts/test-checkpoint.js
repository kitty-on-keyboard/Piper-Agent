#!/usr/bin/env node
// Classify / effectiveDraftDir / sibling MTP discovery, no VS Code host.
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const out = path.join(root, "out/checkpoint.js");
if (!fs.existsSync(out)) {
  execFileSync("npx", ["tsc", "-p", "."], { cwd: root, stdio: "inherit" });
}

const {
  classifyCheckpoint,
  isUsableMtp,
  findSiblingMtp,
  effectiveDraftDir,
  pushRecent,
} = require(out);

function memFs(files) {
  return {
    readText(p) {
      return files[p];
    },
    listDirNames(dir) {
      const prefix = dir.endsWith(path.sep) ? dir : dir + path.sep;
      const names = new Set();
      for (const p of Object.keys(files)) {
        if (!p.startsWith(prefix)) continue;
        const rest = p.slice(prefix.length);
        const top = rest.split(path.sep)[0];
        if (top) names.add(top);
      }
      return [...names];
    },
  };
}

const parent = path.join("/tmp", "lmp-models");
const moe = path.join(parent, "Qwen3.6-35B-A3B-MLX-4bit");
const dense = path.join(parent, "Qwen3.8-27B-MLX-4bit");
const mtp = path.join(parent, "Qwen3.8-27B-MTP-4bit");
const otherMtp = path.join(parent, "unrelated-MTP");
const notMtp = path.join(parent, "Qwen3.8-27B-MLX-4bit-copy");

const io = memFs({
  [path.join(moe, "config.json")]: JSON.stringify({
    model_type: "qwen3_5_moe",
    text_config: { model_type: "qwen3_5_moe_text" },
  }),
  [path.join(dense, "config.json")]: JSON.stringify({
    model_type: "qwen3_5",
    text_config: { model_type: "qwen3_5_text" },
  }),
  [path.join(mtp, "config.json")]: JSON.stringify({
    model_type: "qwen3_5_mtp",
    block_size: 3,
  }),
  [path.join(otherMtp, "config.json")]: JSON.stringify({
    model_type: "qwen3_5_mtp",
    block_size: 3,
  }),
  [path.join(notMtp, "config.json")]: JSON.stringify({
    model_type: "qwen3_5",
    text_config: { model_type: "qwen3_5_text" },
  }),
  [path.join(parent, "tiny-mtp", "config.json")]: JSON.stringify({
    model_type: "qwen3_5_mtp",
    block_size: 1,
  }),
});

assert.strictEqual(classifyCheckpoint(moe, io), "moe");
assert.strictEqual(classifyCheckpoint(dense, io), "dense");
assert.strictEqual(classifyCheckpoint(mtp, io), "mtp");
assert.strictEqual(classifyCheckpoint("/no/such", io), "unknown");
assert.strictEqual(classifyCheckpoint("", io), "unknown");

assert.strictEqual(isUsableMtp(mtp, io), true);
assert.strictEqual(isUsableMtp(dense, io), false);
assert.strictEqual(isUsableMtp(moe, io), false);
assert.strictEqual(isUsableMtp(path.join(parent, "tiny-mtp"), io), false);

assert.strictEqual(findSiblingMtp(dense, io), mtp);

assert.strictEqual(
  effectiveDraftDir(moe, { speculative: true, draftModelDir: mtp }, io),
  "",
  "MoE must never be sent a leftover MTP path"
);
assert.strictEqual(
  effectiveDraftDir(dense, { speculative: false, draftModelDir: mtp }, io),
  "",
  "checkbox off means no draft, even on dense"
);
assert.strictEqual(
  effectiveDraftDir(dense, { speculative: true, draftModelDir: mtp }, io),
  mtp
);
assert.strictEqual(
  effectiveDraftDir(dense, { speculative: true, draftModelDir: "" }, io),
  mtp,
  "empty stored draft still finds the sibling MTP"
);
assert.strictEqual(
  effectiveDraftDir(dense, { speculative: true, draftModelDir: moe }, io),
  mtp,
  "a stored path that is not an MTP head falls through to sibling discovery"
);

assert.deepStrictEqual(pushRecent([], dense), [dense]);
assert.deepStrictEqual(pushRecent([dense], moe), [moe, dense]);
assert.deepStrictEqual(pushRecent([moe, dense], moe), [moe, dense]);
assert.deepStrictEqual(
  pushRecent(["a", "b", "c", "d"], "e", 4),
  ["e", "a", "b", "c"]
);

console.log("test-checkpoint: ok");
