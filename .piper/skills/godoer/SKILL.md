---
name: godoer
description: Use this when working with Godot projects or Godoer MCP to inspect, edit, and verify GDScript, scenes, resources, and project settings.
---

# Godoer Skill Recipe

This recipe guides agent workers when modifying or debugging Godot 4.x game projects using Piper.

## 1. Environment & Tools
- When the `godoer` MCP server is connected (via `trust_mcp: ["godoer"]`), use the structured MCP tools to inspect scene trees, evaluate scripts, and run node checks.
- If MCP is absent, fallback to `exec_command` with headless Godot commands:
  - Check GDScript syntax: `godot --headless --check-only -s path/to/script.gd`
  - Run test harness: `godot --headless -s addons/gut/gut_cmdln.gd` (or project test runner).

## 2. DO NOT TOUCH Rules
- **Never hand-edit `.godot/`**: This directory contains editor cache and imported asset artifacts. Godot regenerates it automatically.
- **Preserve Scene UIDs**: Do not arbitrarily modify `uid://...` hashes in `.tscn` and `.tres` headers; removing or altering them breaks internal project dependencies.
- **Preserve `project.godot` Header**: Only edit project settings sections that you are directly tasked with modifying.

## 3. Slicing Godot Work
- **Single Responsibility per Slice**: Limit edits to one script or one scene per turn.
- **Decouple Logic and State**: Put game data in Custom Resources (`.tres`) and logic in GDScript (`.gd`).
- **Explicit Signal Wiring**: Prefer connecting signals in GDScript `_ready()` (e.g., `button.pressed.connect(_on_button_pressed)`) or verify scene connection bindings before changing handler method signatures.

## 4. Common Pitfalls & Traps
- **Indentation**: Godot standard GDScript uses tabs. Never mix tabs and spaces.
- **@onready vs _ready()**: Do not access child nodes during `_init()`; use `@onready var child = $Child` or fetch them in `_ready()`.
- **Cyclic Class Dependencies**: Using `class_name` across mutually dependent scripts can trigger parse failures during headless validation.
- **Node Paths**: Hardcoded paths like `$"../UI/Label"` break easily during scene refactoring; prefer unique scene names (`%Label`) or `@export var label: Label`.

## 5. Verification Checklist
- Run headless parse checks on all touched scripts.
- Ensure all `@export` references and signals resolve without warnings.
- Run project unit tests with `godot --headless` before concluding the slice.
