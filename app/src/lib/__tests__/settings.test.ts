import { describe, expect, it } from "vitest";
import {
  DEFAULT_SETTINGS,
  mergeLoadedSettings,
  sanitizeAllowedCommands,
  toRunSettings,
} from "../settings";

describe("settings", () => {
  it("produces standalone sidecar RunSettings without editor bridge flags", () => {
    const merged = mergeLoadedSettings({
      workspace_root: "/tmp/repo",
      model_dir: "/models/qwen",
      mode: "plan",
      sandbox_tier: 3,
      allowed_commands: ["pytest", "pytest", "bad\ncmd", "cargo test"],
    });
    const run = toRunSettings(merged);
    expect(run.applies_edits).toBe(false);
    expect(run.provides_code_intel).toBe(false);
    expect(run.sandbox_tier).toBe(0);
    expect(run.allowed_commands).toBe("pytest\ncargo test");
  });

  it("sanitizes multiline and duplicate command allowlist entries", () => {
    expect(sanitizeAllowedCommands([" ls ", "", "a\nb", "ls", "cmake --build"])).toEqual([
      "ls",
      "cmake --build",
    ]);
  });

  it("falls back to Qwen defaults when fields are absent", () => {
    const cfg = mergeLoadedSettings({}, "/fallback/model");
    expect(cfg.model_dir).toBe("/fallback/model");
    expect(cfg.sampling).toEqual(DEFAULT_SETTINGS.sampling);
  });
});
