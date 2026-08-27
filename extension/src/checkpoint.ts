// Classify a Qwen3 MLX folder from config.json alone — no sidecar, no weights.
//
// The editor used to send `draftModelDir` on every load. That path is an MTP head that
// only works on the dense 27B; pointing it at A3B fails the whole load. This module is
// the gate: MoE never receives a draft, and dense only does when the checkbox is on.

import * as fs from "fs";
import * as path from "path";

export type CheckpointKind = "dense" | "moe" | "mtp" | "unknown";

export interface CheckpointFs {
  readText(absPath: string): string | undefined;
  listDirNames(absPath: string): string[];
}

export const nodeCheckpointFs: CheckpointFs = {
  readText(absPath: string): string | undefined {
    try {
      return fs.readFileSync(absPath, "utf8");
    } catch {
      return undefined;
    }
  },
  listDirNames(absPath: string): string[] {
    try {
      return fs
        .readdirSync(absPath, { withFileTypes: true })
        .filter((e) => e.isDirectory())
        .map((e) => e.name);
    } catch {
      return [];
    }
  },
};

interface ConfigJson {
  model_type?: unknown;
  block_size?: unknown;
  text_config?: { model_type?: unknown };
}

function parseConfig(dir: string, io: CheckpointFs): ConfigJson | undefined {
  const raw = io.readText(path.join(dir, "config.json"));
  if (raw === undefined) return undefined;
  try {
    return JSON.parse(raw) as ConfigJson;
  } catch {
    return undefined;
  }
}

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asInt(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

export function classifyCheckpoint(dir: string, io: CheckpointFs = nodeCheckpointFs): CheckpointKind {
  if (!dir) return "unknown";
  const cfg = parseConfig(dir, io);
  if (!cfg) return "unknown";
  // MTP is a root-only model_type. The C++ loader refuses anything else as a draft head
  // (pointing draftModelDir at the target itself is the operator mistake it exists for).
  if (asString(cfg.model_type) === "qwen3_5_mtp") return "mtp";
  const mt = asString(cfg.text_config?.model_type) || asString(cfg.model_type);
  if (mt === "qwen3_5_moe" || mt === "qwen3_5_moe_text") return "moe";
  if (mt === "qwen3_5" || mt === "qwen3_5_text") return "dense";
  return "unknown";
}

export function isUsableMtp(dir: string, io: CheckpointFs = nodeCheckpointFs): boolean {
  if (!dir) return false;
  const cfg = parseConfig(dir, io);
  if (!cfg) return false;
  if (asString(cfg.model_type) !== "qwen3_5_mtp") return false;
  return asInt(cfg.block_size) >= 2;
}

function guessMtpNames(denseBase: string): string[] {
  const guesses = [
    denseBase.replace(/-MLX-4bit$/i, "-MTP-4bit"),
    denseBase.replace(/-MLX-/i, "-MTP-"),
    denseBase.replace(/MLX/gi, "MTP"),
  ];
  return [...new Set(guesses)].filter((g) => g !== denseBase);
}

function sharedPrefixLen(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}

export function findSiblingMtp(
  denseDir: string,
  io: CheckpointFs = nodeCheckpointFs
): string | undefined {
  const parent = path.dirname(denseDir);
  const base = path.basename(denseDir);
  const mtpDirs: string[] = [];
  for (const name of io.listDirNames(parent)) {
    if (name === base) continue;
    const full = path.join(parent, name);
    if (isUsableMtp(full, io)) mtpDirs.push(full);
  }
  if (mtpDirs.length === 0) return undefined;
  for (const guess of guessMtpNames(base)) {
    const hit = mtpDirs.find((d) => path.basename(d) === guess);
    if (hit) return hit;
  }
  mtpDirs.sort(
    (a, b) => sharedPrefixLen(path.basename(b), base) - sharedPrefixLen(path.basename(a), base)
  );
  return mtpDirs[0];
}

export function effectiveDraftDir(
  modelDir: string,
  opts: { speculative: boolean; draftModelDir: string },
  io: CheckpointFs = nodeCheckpointFs
): string {
  if (!opts.speculative) return "";
  if (classifyCheckpoint(modelDir, io) !== "dense") return "";
  if (isUsableMtp(opts.draftModelDir, io)) return opts.draftModelDir;
  return findSiblingMtp(modelDir, io) ?? "";
}

export function pushRecent(prev: string[], dir: string, cap = 4): string[] {
  if (!dir) return prev.slice(0, cap);
  return [dir, ...prev.filter((d) => d && d !== dir)].slice(0, cap);
}
