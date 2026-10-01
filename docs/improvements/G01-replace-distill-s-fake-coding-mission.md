# G01: Replace distill's fake coding mission with a daemon analyze job: one constrained turn on the resident model, no tools, no workspace, no model swap

| Status | Repo | Kind | Impact | Effort |
|---|---|---|---|---|
| VERIFIED (reproduced) | piper | design | high | M |

> **Verified.** The problem and fix below are the skeptic's corrected versions. They replace the finder's original claim where the two differ.

Merged from these finder reports:

- service-design#1 Daemon analyze job replaces distill's fake coding-task packet
- distill-piper#3 Warm distill runs a full auto-approve-all agent mission
- distill-godoer#4 Distill should be a tool-less, schema-constrained diagnosis (Piper half)
- ideas-godoer#1 One shared analyze path (Piper transport: analyze method, single-tool turn)
- distill-godoer#6 Godoer duplicates Piper's model/python discovery (warm-path model swap)
- distill-piper#6 Docs document a distill CLI that does not exist

## Problem

Warm `piper distill` sends each incident to the daemon as a full Agent-mode coding mission. The packet sets no mode, so Agent policy applies: T1 sandbox, exec, writes and destructive ops. The model gets the "You are changing this codebase" brief and kDefaultMaxIterations=30, with timeout_s=300. All approvals are on: the packet sets exec, writes and irreversible, and the wire adds auto_approve_all. cwd is <project>/.godoer, which holds Godoer's control-plane state (gate_audit.jsonl, last_play.json, watch.lock, incidents.json, assets.json).

CORRECTION: the game sources cannot be edited. In Agent mode the shell runs at T1 Seatbelt, which denies file-write* outside the workspace root (src/tools/sandbox.cpp:103-130), and file tools are confined by fs.cpp. Reads are open, so the shell can only cat the res:// source. The real write hazard is the whole .godoer/ directory, which the agent can write to and delete from with no approval. Every warm distill also leaves .lmp-context.db* (the PCC journal, sidecar.cpp:1306; later distills are then told recall exists) and .lmp_tmp/ inside .godoer/.

The per-slice event log and live.jsonl go to the temp dir, which is then deleted, not into the game repo. collect_git does run `git diff` and `git status` over the game's work tree for every incident.

All other claims hold:
- **Model swap:** model_dir is forced (args, then LMP_QWEN_DIR, then the /Users/dev 27B path). Godoer passes find_local_model(), which tries the 27B first. ensure_model reloads whenever the names differ. This only happens on a mismatch, but README lists the 35B-A3B as the primary model.
- **Prompt:** the hand-built ChatML is inert user text under the coding system prompt on the warm path, and a raw template on the mlx_lm cold path.
- **Limits:** --max-tokens is dropped on the warm path. The answer is capped at 2000 chars, or cut to 480-char bookends when the run stalls.
- **Failures:** the daemon reply, including load_packet errors, is discarded. The failure is then filed as a diagnosis with status 'diagnosed' and exit 0.
- **Docs:** the documented commands exit 3. The "zero RAM" and "system-wide lock" claims are false (the lock is cold-only).
- **is_parent_harness_command:** it lacks 'distill', but this only matters when the C++ binary is invoked directly. Bare `piper` is a symlink to piper_worker.py.

Additional facts found:
- Both Python liveness probes (piper_worker._is_daemon_alive and godoer is_piper_daemon_alive) report a BUSY daemon as offline. With --allow-cold (Godoer's force path or GODOER_AUTO_DISTILL=1), that cold-loads a second model next to the resident one. The C++ probe already treats busy as alive (socket_reader.cpp:83-89).
- Godoer's MCP from_run calls auto_distill synchronously with a 180 s timeout. A failed run's reply can block for 180 s while the daemon keeps running the orphaned mission, delaying the next dispatch.
- Godoer's warm path crashes when find_local_model() is None: `diag_data.get("model", Path(model_dir).name)` builds its default eagerly. That is why test_auto_distill_daemon_online_invokes_distill fails on any host without the hardcoded models.

## Why it matters

This is the feature the user likes most. Today it hands a write-capable, auto-approved 30-turn coding agent a crash log inside their game repo. It can evict the checkpoint their cloud-dispatched slices run on, and it reports its own failures as diagnoses. Every other distill improvement (structured card, deadlines, busy handling, new card kinds) needs this transport first.

## Fix to ship

Ship the analyze method with these changes, as one change across both repos.

PIPER C++
1. New src/surface/analyze.{hpp,cpp}:
   - parse_analyze_request with strict validation: proto==1, job=="incident_diagnosis" (the only entry), 1-32 items, source_context ≤8 KB, deadline_s >0, max_new_tokens clamped.
   - A constexpr system brief plus one parsephony ToolSpec, report_diagnosis (fields per G05).
   - render_item_user(item), built deterministically from the item fields.
   - run_analyze(req, tok, backend, cancel, emit), per item:
     - Render the prompt with ChatTemplate::render({System brief, User item}, tools_json).
     - Mask chain: TurnGrammar(tok, {spec}) → ThinkCapMask → a new small MustCallMask (same pattern as ToolCapMask) that removes <|im_end|> until grammar.has_tool_call().
     - Generate through GrammarSink and backend.generate.
     - Map GenStatus to item status ok|length_capped|cancelled|error, and the parsed params to the result.
   - No Agent, registry, MCP, ContextJournal, workspace, result.json, git or PLAN.md.
2. sidecar.cpp serve loop gets `method=="analyze"`:
   - Drop the request if the peer is already gone (poll plus recv MSG_PEEK==0).
   - Model: use session.backend/tok if resident and NEVER call ensure_model on a client value. Otherwise load DaemonConfig.model_dir: new `serve --model-dir`, else LMP_QWEN_DIR captured at serve start. Otherwise reply {status:no_model}.
   - Write {kind:accepted}, then run the items.
   - A watchdog jthread cancels at deadline_s or on client POLLHUP.
   - Stream one item line per item, then {status ok|partial|cancelled|error|no_model, model_dir, wall_ms}.
   - Write one audit line per item to a daemon log.
3. worker.cpp: add "distill" to is_parent_harness_command.
4. Cold path:
   - Delete the mlx_lm cold path.
   - If cold is kept at all, it becomes `lmp_sidecar --worker analyze` one-shot, running the same run_analyze so the prompt and grammar match the warm path. Gate it with the C++ is_daemon_alive, which counts busy as alive.

PIPER PYTHON (scripts/piper_worker.py cmd_distill becomes a thin client)
- Keep --incidents, --out and --socket; add --deadline-s. Drop --model-dir and --max-tokens plumbing, the /Users/dev defaults, the temp task dirs, the ChatML and /tmp/piper_distill.lock.
- Map the connection outcome to a report status:
  - no socket or refused → skipped_daemon_offline
  - connected but no `accepted` within ~2 s → busy, exit 0, never cold
- Report status is 'diagnosed' only when every item is ok; otherwise partial, busy, no_model, cancelled or error.
- Only ok items carry a diagnosis; other items carry their status and error.
- Delete or fix _is_daemon_alive so a timeout never means offline.
- Add self-test scenarios with a fake daemon for: ok, partial, error, no_model, busy, request carries no auto_approve or model_dir fields, and the documented command lines.

GODOER (godoer/telemetry/distiller.py)
- Drop find_local_model and find_mlx_python from the warm path and never pass --model-dir.
- Pass --deadline-s = timeout − 10.
- Read report.status and item status. Add DistillationResult statuses busy, no_model, partial and error; never 'ok' carrying failure text.
- Remove the eager Path(model_dir).name.
- Update test_auto_distill_daemon_online_invokes_distill (assert no --model-dir) and add status-mapping tests.

DOCS
- Fix PIPER.md and both piper-orchestration SKILL.md copies: real flags and statuses, drop the zero-RAM and lock claims, no stdin until a raw-log mode exists (which would feed log_triage unchanged).
- Keep a single job; no generic job framework, no --json.

## Evidence (file:line)

- piper:scripts/piper_worker.py:1284-1294 task_data: cwd=dirname(incidents) (= project/.godoer), auto_approve_exec/writes/irreversible True, model_dir forced, timeout_s 300, no token limit
- piper:scripts/piper_worker.py:1143-1148 the socket request adds auto_approve_irreversible and auto_approve_all; src/surface/sidecar.cpp:2574-2576 applies them to the packet
- piper:src/surface/worker.hpp:39 mode defaults to 'agent'; :78 kDefaultMaxIterations = 30; src/loop/turn.cpp:111-112 Agent policy grants exec+write; :288 brief 'You are changing this codebase'
- piper:src/surface/worker.cpp:462-472 build_start_message: a full mission (workspace_root=cwd, auto-approve flags, require_approval=false)
- piper:src/platform/fs.cpp:94-103 absolute paths outside the workspace root are rejected for file tools, so res:// sources are reachable only through the auto-approved shell
- godoer:godoer/cli.py:1520-1529 and godoer/gate/export.py:1 gate_audit.jsonl lives in that same .godoer/ directory
- piper:scripts/piper_worker.py:1202 model_dir = args.model_dir or LMP_QWEN_DIR or '/Users/dev/Desktop/Models/Qwen3.8-27B-MLX-4bit'; godoer:godoer/telemetry/distiller.py:31-34,72-82,217,227-228 Godoer resolves its own model (27B listed first) and passes --model-dir on the warm path too
- piper:src/surface/sidecar.cpp:840-850 ensure_model loads when !session.holds(model_dir); src/surface/session.cpp:45-52 the resident weights are released first; .cursor/skills/piper-orchestration/SKILL.md:8 the worker runs 'typically Qwen3.6-35B-A3B-MLX-4bit'
- piper:scripts/piper_worker.py:1253-1278 hand-built '<|im_start|>system ... <|im_start|>assistant' prompt; src/model/chat_template.hpp:16-18 message content goes through encode_content, so those literals are inert user text; scripts/piper_worker.py:1311-1319 the cold path passes the same string to mlx_lm as a raw template
- piper:src/surface/sidecar.cpp:1937-1941 opens an event log, and :2298-2307 runs collect_git/collect_git_changed_paths on the packet cwd, for every incident
- piper:scripts/piper_worker.py:1298 the forward reply is discarded; :1304-1308 the result message or error, else 'Daemon finished but no result.json was produced', becomes the diagnosis; :1352-1357 report status is always 'diagnosed'
- piper:src/surface/worker.cpp:901-921 the result message is capped at 2000 chars, with 480-char bookends when the run is incomplete
- piper:PIPER.md:118,121 show `piper distill --input ...` and `cat engine_run.log | piper distill`; :124-126 claim 'zero memory allocation' and a system-wide lock. .agents/skills/piper-orchestration/SKILL.md:82 says the same, and diff shows the .cursor copy is identical. scripts/piper_worker.py:2454 makes --incidents required. Both documented forms re-run here: exit 3
- piper:src/surface/worker.cpp:1640-1645 is_parent_harness_command has no 'distill'
- piper:src/model/backend.hpp:75 InferenceTask, :112-114 GenStatus including LengthCapped, :198 generate(task, sink, cancel); src/model/grammar.hpp:64 TurnGrammar. The primitives a one-shot job needs already exist

## Skeptic verdict

real=True, reproduced=True, keep=True, root-cause fix=True, already handled=False, violates constraint=False

VERIFIED BY READING:
- Every cited file:line is accurate:
  - piper_worker.py 1139-1373 and 2449-2459
  - sidecar.cpp:
    - 2526-2588: serve loop; 'run' applies auto_approve_all
    - 1907-2367: execute_task_packet
    - 840-872: ensure_model
    - 1025-1060: start_mission
  - worker.cpp:
    - 164-446: load_packet (requires model_dir to be a dir)
    - 462-572: build_start_message (mission = prompt)
    - 891-941: compose_result_message
    - 1640-1645: is_parent_harness_command
  - worker.hpp:39,78; turn.cpp:97-118 and 280-292; fs.cpp:94-103; chat_template.hpp:16-19; backend.hpp:75/112/198; grammar.hpp:64 (ThinkCapMask, ToolCapMask; GrammarSink and LoopBreaker in loop/token_stream.hpp)
  - godoer distiller.py:31-34,72-82,217-228; cli.py:1520 and gate/audit.py:38 for gate_audit.jsonl; watch.py:105 for watch.lock
- The refutation of "shell can edit the game": ModePolicy::for_mode(Agent)={tier 1,...}; nothing in the worker path sets sandbox_tier; seatbelt_profile = allow default + deny file-write* + allow the workspace subpath.
- Only Agent::step calls backend.generate today, so a one-shot path must be built. The primitives exist.

REPRODUCED HERE (scratchpad fake daemon on a unix socket, real scripts/piper_worker.py):
- (a) The daemon replies {status:error, error:'model_dir is not a directory'} exactly as sidecar.cpp:2562-2572 does. The report comes back status=diagnosed, exit 0, with every incident's diagnosis = 'Daemon finished but no result.json was produced'.
- (b) The daemon writes a stalled result.json. The agent's narration is filed as the diagnosis and the stall is lost.
- The captured task.json shows:
  - cwd=<proj>/.godoer, no mode
  - exec, writes and irreversible all True; the request adds auto_approve_all
  - model_dir=/Users/dev/.../Qwen3.8-27B-MLX-4bit, timeout 300, no token limit
  - prompt beginning '<|im_start|>system'
- The documented `piper distill --input` and stdin forms exit 3.
- A listener that is not accepting (a busy daemon) makes both Python probes return False after 1 s.
- Godoer tests (pytest in a scratch venv): 4 pass. test_auto_distill_daemon_online_invokes_distill FAILS because of the Path(None) crash. A probe shows that with a model path present, a failure string comes back as status 'ok'.
- A queued request from a client that gave up is still read by the server later, and MSG_PEEK reports the peer gone. So a daemon-side 'busy' reply cannot work.
- piper_worker self-test has zero distill scenarios.

VALUE AND DESIGN. I tried to kill it and could not:
- Reusing 'run' with mode plan/debug and a small max_iterations is a band-aid. Plan mode is conversational (ask_user then a 300 s wait) and writes PLAN.md into cwd. The workspace, PCC journal, registry, git collection and the 2000-char finish text all remain.
- Running a Python mlx_lm call would put a second model in RAM, which violates the one-model constraint.
- The candidate's design strengthens one-model, adds no subagents, removes harness git from distill, and touches no measured engine.
- A local 27B/A3B can reliably write one grammar-constrained diagnosis from ±15 lines of deterministic source context.
- It also helps orchestration: no distill-induced reloads, and a dispatch waits at most deadline_s instead of up to 300 s × incidents.

Flaws in the candidate's fix:
- (1) 'busy' as a daemon reply is impossible. The serve loop is single-threaded and does not accept while working. Busy has to be detected by the client.
- (2) The cold path is unspecified. Keeping mlx_lm keeps different prompts and no grammar.
- (3) Nothing forces the report call. TurnGrammar lets the turn end in prose.
- (4) The Godoer half is not listed, but the wire change redefines its statuses.
- (5) --json is churn; the --out file is the contract.
- (6) The job table should hold one job, not a framework.

Minor: an interleaved analyze resets the backend KV, so the next dispatch prefills its prompt once. This is negligible and not perf work.

Back to the [index](README.md).
