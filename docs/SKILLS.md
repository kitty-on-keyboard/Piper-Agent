# Skills System in Piper

Piper supports Cursor/Claude-style skill recipes so the local agent can discover and load specialized, reusable recipes on demand without permanently bloating the stable prompt prefix.

## 1. On-Disk Format

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: godoer
description: Use this when working with Godot projects or Godoer MCP to inspect, edit, and verify GDScript, scenes, resources, and project settings.
---

# Godoer Skill Recipe

... instructions, guidelines, verification steps ...
```

### Frontmatter Schema
- `name`: Short human-readable display name.
- `description`: Clear trigger condition explaining *when* and *why* the agent should load this skill.

### Caps & Safety Constraints
- **Skill IDs**: Must match `^[A-Za-z0-9_-]+$`. Path traversal (`..`, slashes) is strictly rejected.
- **Frontmatter**: Maximum 4 KiB parsed between opening and closing `---` markers.
- **Skill Body**: Maximum 64 KiB. Files exceeding this budget are truncated with an explicit warning note.

---

## 2. Discovery Roots & Precedence

When scanning for skills, Piper scans the following allowlisted roots in strict priority order (first match wins):

1. **`<cwd>/.piper/skills/<id>/SKILL.md`** — Piper-specific workspace skills.
2. **`<cwd>/.cursor/skills/<id>/SKILL.md`** — Cursor-compatible workspace skills.
3. **`<cwd>/.agents/skills/<id>/SKILL.md`** — Standard agent workspace skills.
4. **`~/.piper/skills/<id>/SKILL.md`** — User-global skill library.

If a skill with the same `id` exists in both `.piper/skills` and `.cursor/skills`, the earlier root takes precedence and shadows subsequent locations.

---

## 3. Agent Tools

Piper exposes two dedicated tools for skill management:

### `list_skills`
Scans the allowlisted roots and returns the catalog of all available skills:
```
Available skills (2):
- godoer (godoer) [.piper/skills]: Use this when working with Godot projects...
- synth (Audio Synth) [~/.piper/skills]: Use this when writing DSP/C++ synth nodes...
```

### `load_skill(id: string)`
Loads the full recipe body of the requested skill into the active session. The recipe is returned in the tool response and permanently injected into the session's prompt context under `# Loaded skills` for the duration of the run.

---

## 4. Task Packet Preloading

For headless worker slices (e.g. dispatched by cloud orchestrators or CI), skills can be preloaded before turn 0 so the model starts with the recipe already in context.

In `task.json`:
```json
{
  "id": "slice-042",
  "cwd": "/path/to/project",
  "model_dir": "/path/to/model",
  "prompt": "Fix player jump gravity in Player.tscn",
  "skills": ["godoer"]
}
```

Preloaded skills:
- Bypass the need for an initial `load_skill` tool call turn.
- Are loaded into `# Loaded skills` immediately on `lmp/start`.
- Emit `skill_loaded` events into the structured event log with `preload: "1"`.

---

## 5. Prompt Context Structure

In `ContextStore::render()`, skills are organized into two sections:

1. **`# Available skills`**: Rendered whenever skills are discovered in the workspace or user library. Provides the model with a compact sitemap of available skills and their descriptions.
2. **`# Loaded skills`**: Rendered only when skills have been loaded (via `load_skill` or task packet preloading). Contains the complete markdown recipe body.
