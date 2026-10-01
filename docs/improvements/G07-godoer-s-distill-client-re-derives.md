# G07: Godoer's distill client re-derives Piper's internals and trusts any file on disk: successes read as errors, skips and stale cards read as ok, failed runs stale evidence

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | both | bug | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- distill-godoer#2 Diagnosis artifacts never tied to the run that produced them
- distill-godoer#6 Godoer duplicates Piper's model/python discovery (Godoer-side duplication, Path(None) crash)
- distill-godoer#5 Telemetry files in .godoer/ count as project input
- distill-piper#4 Diagnosis card is unstructured (Godoer report-contract bugs: ignored status, Path(None), mocked boundary)

## Problem

All seven sub-claims are true. How often each one fires differs from the headline.

(a) godoer/telemetry/distiller.py:264 `diag_data.get('model', Path(model_dir).name)` raises whenever find_local_model() is None. A real 'diagnosed' report then comes back as status 'error'. Reproduced through the real scripts/piper_worker.py distill against a fake daemon. It does not fire on the user's Mac, where the /Users/dev model folders exist, so the test passes there. tests/test_telemetry_distiller.py is not in .github/workflows/evidence.yml, which is why it is red only on other machines.

(b) Godoer ignores the report's status. A skipped_daemon_offline report becomes 'ok' with summary '' and a model label Godoer made up. It can only happen through a probe race against the single-threaded daemon: Godoer's ping succeeds, then Piper's 1-second ping fails because a run has just started. Every caller prints only dist.summary, so the visible effect is a hint that silently disappears.

(c) A stale output file is returned as 'ok'. Reproduced end to end: a new player.gd:164 fatal returned the old OLD.gd:99 card under the same INC-001. However, Piper's exit-0 path that writes nothing is in cold mode only, and Godoer only goes cold when the undocumented GODOER_AUTO_DISTILL=1 is set (no caller passes force=True). The same fixed paths also let two concurrent failing runs on one project overwrite each other's incidents.json, in warm mode too.

(d) Common, and the most visible. Fail, distill, fix, then a green play: `godoer play --report` (play_digest.py:165-178) prints "ok": true followed by the old card. incidents.json is never cleared. Godoer's skills/godoer/SKILL.md:95,115 and Piper's .piper/skills/godoer/SKILL.md:12,37,42 both tell agents to read it, the Piper one "to verify no runtime script errors were triggered". The Piper skill also wrongly says the file holds timestamps, node paths and stack traces.

(e) Wider than stated. Every failing run writes .godoer/incidents.json, even with no Piper installed (the skipped_no_piper path). evidence.verify defaults to current_inputs=True, so verdicts become inconclusive. This breaks SKILL.md's documented rule that bookkeeping alone does not make evidence stale. .godoer/gate_audit.jsonl, which the gate proxy appends on every gated tool response that carries a project argument, has the same defect. The root cause is the hand-maintained EXCLUDED_PATHS deny-list, not telemetry alone.

(f) True, and find_mlx_python is dead code. But removing --model-dir alone does not stop daemon model swaps. With no --model-dir, Piper's cmd_distill still put '/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit' into the daemon task (observed), and Session::holds compares strings exactly. The swap fix belongs to Piper (G01).

(g) True. On top of that, the test is not run in CI at all.

## Why it matters

The card is the product. A stale card labelled 'ok' sends a cloud agent to fix a bug that is already fixed, or the wrong one, and the status inversions mean nobody can trust the result field. Three of these bugs are reproduced on current code, and one keeps the repo's own test red. The evidence staling silently forces re-evaluation after the most common event, a failed run.

## Fix to ship

Ship the Godoer half now; it is mostly deletion. Piper needs about 10 lines.

Godoer:
1. godoer/telemetry/distiller.py
   - Delete _DEFAULT_MODELS, _DEFAULT_MLX_PYTHONS, _DEFAULT_PIPER_PATHS, find_local_model and find_mlx_python, along with their exports in telemetry/__init__.py.
   - find_piper_worker becomes PIPER_WORKER_PATH, else shutil.which('piper') resolved. install_piper_link.sh already symlinks `piper` to scripts/piper_worker.py.
   - Never pass --model-dir.
   - For each invocation, write the incidents packet into a fresh tempfile.mkdtemp() outside the project, and pass an --out path there that does not exist yet.
   - If no report is written, the status is 'error'.
   - Map the report status one to one: 'diagnosed' becomes ok; 'skipped_daemon_offline' keeps Godoer's existing hint text; an unknown status or a malformed report is 'error'.
   - The model label is report.get('model') or ''. This also removes the Path(None) crash.
   - Only a valid 'diagnosed' report gets promoted, with os.replace, to .godoer/telemetry/diagnosis.json. Stamp each diagnosis with its incident key. Matching on incident_id is safe within one isolated invocation, so Piper does not need to echo anything back.
2. godoer/telemetry/extractor.py: add Incident.key = sha256(kind, category, file, line, message). This is the site identity only: no source window, and no last_play hash.
3. play_digest.report_digest: render only diagnoses whose key matches a fatal in the current last_play.json. A green digest then shows no card, and old files without keys are ignored.
   - Stop treating incidents.json as a second, persisted source of truth. --report derives incidents from last_play.json, which is deterministic and cheap. Both SKILL.md files (Godoer, and Piper's .piper/skills/godoer) point at `godoer play --report`. Also remove the false timestamps, node paths and stack traces claim.
4. godoer/evidence.py: add '.godoer/telemetry' and '.godoer/gate_audit.jsonl' to EXCLUDED_PATHS.
   - Add a behavioural test to tests/test_evidence.py: a failing auto_distill (no Piper) and gate.audit.append_audit must both leave tree_fingerprint unchanged.
   - Update the SKILL.md line about evidence.
5. Tests:
   - Replace the patched subprocess.run with a fake `piper` executable set through PIPER_WORKER_PATH. Cover each status, exit 0 with no report, and a stale file already present.
   - Add the telemetry and play_digest tests to .github/workflows/evidence.yml; they are pure Python.
   - Keep the real-harness test with a fake daemon local only, skipped unless PIPER_WORKER_PATH is set.

Piper (scripts/piper_worker.py, cmd_distill):
- Lock contention writes {"status": "skipped_busy"} instead of a bare EXIT_OK.
- The daemon path stops injecting a default model_dir (G01).
- Per-diagnosis status and a 'partial' status follow from G05. Godoer already treats unknown statuses as errors, so it stays compatible.

Defer: reuse diagnoses keyed by (key, hash of the source window). Today every failing run or MCP reply blocks for up to 180s on distill. Once keys exist, this is about 20 lines, but it is a separate feature.

Drop from the candidate:
- The run_id taken from the last_play hash.
- Deleting telemetry on any green run.
- Putting source_context in the join key.
- A contract test that needs a Piper checkout in Godoer CI.

## Evidence (file:line)

- godoer:godoer/telemetry/distiller.py:264 diag_data.get('model', Path(model_dir).name) evaluates the default eagerly and raises when model_dir is None
- godoer: ran tests/test_telemetry_distiller.py with the scratchpad venv: test_auto_distill_daemon_online_invokes_distill fails at :120 (status 'error'), 1 failed / 4 passed; :116-119 the test patches find_piper_worker, is_piper_daemon_alive and subprocess.run
- godoer:godoer/telemetry/distiller.py:261-280 reads only diagnoses and model, ignores diag_data['status'] and always returns status='ok'; :218 the fixed out_file is never cleared before :233; :254 success = out_file.is_file()
- piper:scripts/piper_worker.py:1233-1238 lock contention prints to stderr and returns EXIT_OK without writing --out
- godoer:godoer/telemetry/extractor.py:117-122 dedups by (file, line, category), then sets inc_id = f'INC-{counter:03d}': positional, no content identity
- godoer:godoer/play_digest.py:165-178 renders .godoer/distilled_diagnosis.json whenever it exists, with no check against last_play.json
- godoer:godoer/cli.py:1830-1836, godoer/watch.py:632-640 and godoer/mcp_server.py:184-190 call auto_distill only when not result.ok, so a green run never clears telemetry
- godoer:godoer/evidence.py:32-36 EXCLUDED_PATHS = {.godoer/watch, .godoer/watch.lock, .godoer/last_play.json}, with the comment 'Other .godoer content ... remains input'; :88-104 the walk hashes every other file
- godoer:godoer/telemetry/distiller.py:31-42 hardcoded /Users/dev model, worker and mlx-python lists; :217,227-228 pass --model-dir whenever find_local_model() resolves; skills/godoer/SKILL.md:95,115 tell agents to read .godoer/incidents.json
- Reproduced here (scratchpad venv):
- a valid report with find_local_model None gives 'error: expected str, bytes or os.PathLike object, not NoneType';
- a Piper report of skipped_daemon_offline gives status 'ok' with summary '';
- a leftover diagnosis plus a piper that exits 0 without writing gives 'ok' citing ('res://OLD.gd', 99);
- tree_fingerprint differs after auto_distill, with '.godoer/incidents.json' newly fingerprinted

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

I traced every cited line and all of them hold. Both repos are unmodified. Scratch files are under <scratch, not kept>/g07/.

Evidence:
- pytest in the scratchpad venv: 1 failed, 4 passed, failing at test_telemetry_distiller.py:120.
- repro_a.py: (a), (b) and (c) through Godoer with subprocess mocked.
- repro_de.py: report_digest renders the old card next to "ok": true. tree_fingerprint gains .godoer/incidents.json even on the skipped_no_piper path.
- e2e.py with fake_daemon.py: unpatched auto_distill runs the real piper_worker.py distill against a single-threaded fake daemon, in four scenarios: normal, model_env, race and stale.
- e2e_cold.py with shim/sitecustomize.py: simulates lock contention inside the child process, so nothing under /tmp is touched. Result: status 'ok' with the OLD.gd:99 card.
- I searched both GitHub repos for open PRs. Only the original feature PRs (#270, #252) exist, both closed, so this is not already handled.

Arguments against, and why they don't kill it:
- (a), (b) and (c) are latent or rare on the user's machine.
- (d) and (e) are common. They affect the card, which is the product, and Godoer's evidence verdicts for every user, Piper or not.
- The fix is mostly deletion.
- No constraint is violated. It removes a source of model swaps and needs no C++ change.
- local_model_can_do_it is not applicable: this is a bug, and no runtime model capability is involved.

Flaws in the proposed fix:
- The run_id is defined as the hash of the last_play digest, but `godoer run` and MCP callers pass run_result and have no digest.
- Putting source_context in the join key conflates matching a run with checking that a cached diagnosis is still valid.
- "Cleanup on green" would delete a play-session fatal after a green headless run of a different scene.
- Moving telemetry into an excluded folder misses gate_audit.jsonl, which has the same defect.
- A contract test that needs a Piper checkout cannot run in Godoer CI.
- Removing --model-dir does not stop the swap.

Out of scope, noted only: Piper's distill forwards a full agent task with writes and exec auto-approved. Its workspace is the incidents file's folder (worker.cpp:219-238 takes cwd from the packet). Also, PIPER.md documents `--input` and stdin, which the parser does not support.

Back to the [index](README.md).
