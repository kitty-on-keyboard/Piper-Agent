#pragma once
//
// Worker mode: headless task dispatch, one packet in, one result.json out.
//
// The extension speaks lmp/* over stdio and is driven by a human. Worker mode
// speaks the SAME protocol to the SAME session/loop, but the producer is a
// task.json file and the consumer is a result.json file. One agent, two
// interfaces (worker CLI).
//
// Why this exists instead of a Python wrapper: the sidecar already owns the
// model, the loop, the approval flow and the watchdog. The Python driver
// (scripts/piper_worker.py) was a reimplementation of all of those over
// subprocess stdin/stdout pipes, and keep-warm (holding the model resident
// across slices) cannot be expressed through a process it kills after every run.
//
#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "src/tools/shell_clock.hpp"

namespace lmp::surface::worker {

// Exit codes matching the CLI contract.
inline constexpr int kExitOk = 0;
inline constexpr int kExitError = 1;
inline constexpr int kExitTimeout = 2;
inline constexpr int kExitInvalid = 3;

// ------------------------------------------------------------------
// Packet: what the orchestrator hands us.
// ------------------------------------------------------------------

struct TaskPacket {
    std::string id;
    std::string cwd;
    std::string prompt;
    std::string model_dir;
    std::string mode = "agent";           // "agent" | "plan" | "debug"
    double timeout_s = 900.0;
    bool auto_approve_exec = true;
    bool auto_approve_writes = true;
    bool auto_approve_irreversible = false;
    bool commit_think = true;
    bool shadow_compact = true;
    std::string result_path;
    std::string task_path;                // resolved absolute path of task.json
    std::string task_dir;                 // dirname of task_path

    // Build-out 3: optional check command to run after mission completes.
    // When set, this is both verify_contract (during the loop) and the post-run
    // operator acceptance in result.test. A green post-run check completes the
    // slice (status=ok) even if the agent loop hit max_turns without completed=true.
    std::string check_command;
    // Same clock as the in-loop run of the same command (src/tools/shell_clock.hpp).
    double check_timeout_s = ::lmp::tools::kShellWallClockSeconds;

    // Turn budget for lmp/start. 0 = use resolve_max_iterations() default:
    // kDefaultMaxIterations (30), or kTrustMcpDefaultMaxIterations (60) when
    // trust_mcp is non-empty. Packet field overrides either default.
    int max_iterations = 0;

    // Operator-owned MCP trust voucher (operator MCP trust).
    std::vector<std::string> trust_mcp;

    // Cloud orchestrator webhook URL (orchestrator wake).
    std::string orch_webhook;

    // Skills to preload into the context store on start (by id).
    std::vector<std::string> preload_skills;
};

// Parse task.json (+ optional sibling prompt.md). Returns the packet on success
// or nullopt with `error` filled. Does NOT touch the model or the sidecar.
[[nodiscard]] std::optional<TaskPacket> load_packet(const std::string& path,
                                                    std::string& error);

// Default turn budgets when task.json omits max_iterations.
inline constexpr int kDefaultMaxIterations = 30;
inline constexpr int kTrustMcpDefaultMaxIterations = 60;

// Resolve the turn budget: explicit packet.max_iterations wins; otherwise 30,
// or 60 when trust_mcp is set (Godoer-heavy slices need more turns).
[[nodiscard]] int resolve_max_iterations(const TaskPacket& packet);

// Build the lmp/start JSON-RPC message from a TaskPacket.
[[nodiscard]] std::string build_start_message(const TaskPacket& packet,
                                              const std::string& request_id = "1");

// ------------------------------------------------------------------
// Result: what we hand back.
// ------------------------------------------------------------------

struct DiffStat {
    int insertions = 0;
    int deletions = 0;
    int files = 0;
};

// What the measured log_triage engine (src/tools/log_triage.hpp) found in a red
// check's output: deterministic, no model call.
struct CheckDiagnostic {
    std::string path;                     // as printed; empty when none
    int line = 0;                         // 0 when unknown
    std::string message;
};

struct CheckTriage {
    std::string runner;                   // pytest | ctest | cargo | swift | xcode | unknown
    int passed = -1;                      // -1 = not reported
    int failed = -1;
    std::vector<std::string> failing_tests;   // <= 6
    std::vector<CheckDiagnostic> primary;     // <= 4
    std::vector<std::string> paths;           // <= 8
};

struct TestBlock {
    bool ran = false;
    int exit_code = -1;
    std::string command;
    // log_triage::compact digest of the captured output (<= 2000 bytes). Output that
    // already fits comes back byte for byte.
    std::string output_tail;
    bool timed_out = false;               // killed at check_timeout_s
    bool could_not_run = false;           // never executed (exit 126/127, spawn failure)
    double seconds = 0.0;                 // wall time of the check
    std::string output_path;              // full capture (check.log) when the digest dropped any
    std::optional<CheckTriage> triage;    // red checks only
};

struct RunResult {
    std::string task_id;
    std::string status;                   // "ok" | "error" | "timeout" | "stalled"
    std::string message;                  // think-leak stripped; incomplete runs trimmed
    std::string cwd;
    std::string model_dir;
    double wall_seconds = 0;
    int turns = 0;
    int generated_tokens = 0;
    std::vector<std::string> files_touched;
    DiffStat diff_stat;
    std::string git_diff_path;            // empty when no diff
    TestBlock test;
    std::string log_path;
    std::string error;                    // empty on success
    // How the agent loop stopped (RunReport::termination_reason), and whether the
    // status rests on the operator check alone because the loop itself did not finish.
    std::string termination_reason;
    bool promoted_by_check = false;
    // Tier-A loop hygiene copied from RunReport when the worker path has one.
    std::size_t degenerate_text_count = 0;
    std::size_t text_only_turns = 0;
    std::size_t tool_error_count = 0;
    std::size_t nudged_loop_cut = 0;
    std::size_t nudged_no_progress = 0;
    std::size_t nudged_no_tool_recovery = 0;
};

// What the run did, gathered once at the end of execute_task_packet. Facts only:
// finalize_run turns them (plus the check reading) into the verdict.
struct RunFacts {
    std::string termination_reason;       // RunReport::termination_reason; "" if none
    bool completed = false;               // RunReport::completed
    bool started = true;                  // the mission produced a run at all
    bool irreversible_unanswered = false; // an irreversible ask ended without an answer
    std::string irreversible_detail;      // the command that was held
    double timeout_s = 0.0;               // packet.timeout_s, for the error text
};

struct Finalized {
    std::string status;                   // "ok" | "error" | "timeout" | "stalled"
    std::string error;                    // empty when status is "ok"
    int exit_code = kExitError;
    std::string wake_kind;                // "done" exactly when status is "ok", else "stalled"
    bool promoted_by_check = false;       // ok only because a green check vouched for it
};

// THE one place that decides status, exit code and wake kind. Ordered policy:
//   1. not started                         -> error, exit 1
//   2. timeout_awaiting_user / wall_clock  -> timeout, exit 2, never promoted
//   3. irreversible ask left unanswered    -> error, exit 1, never promoted
//   4. completed or plan_ready             -> ok; a red check demotes it to error
//   5. max_turns, stalled, ended && !completed (incomplete loop stops)
//                                          -> ok when the check is green (promoted),
//                                             else stalled (max_turns/stalled) or error
//   6. anything else (backend_error, cancelled, loop_exit, no run end,
//      stalled_no_turn)                    -> error/stalled, exit 1, never promoted
// Pure: no I/O, so every row is covered by a gate test.
[[nodiscard]] Finalized finalize_run(const RunFacts& facts, const TestBlock& test);

// Why a ran-but-not-green check failed, in words: "check timed out after 300s",
// "check could not run (exit 127)" or "check failed (exit 1)".
[[nodiscard]] std::string describe_check_failure(const TestBlock& test);

// Write result.json atomically (write .tmp, rename).
void write_result(const std::string& path, const RunResult& result);

// Before a new worker run opens the event log beside result.json, copy any existing
// events.jsonl / events.ndjson to events-<UTC>.jsonl so bakeoff retries keep an
// immutable try artifact (e.g. bowling seed7 A/B). No-op when neither file exists.
void archive_prior_events(const std::string& result_dir);

// Prefer events.jsonl beside result.json; keep events.ndjson as a compatibility name.
[[nodiscard]] std::string durable_events_path(const std::string& result_dir);

// ------------------------------------------------------------------
// Helpers that build-outs 2 and 3 need.
// ------------------------------------------------------------------

// Strip <think>...</think> tags and leading preamble ("Let me verify...",
// "I'll now...") from answer text so the orchestrator reads a clean summary.
[[nodiscard]] std::string strip_think_leak(const std::string& answer);

// Collect files touched from the event log: kind=write (changed≠0), plus
// path-like args on tool_call events (MCP/Godoer often skip the write ledger).
[[nodiscard]] std::vector<std::string> collect_files_touched(
    const std::string& log_path, const std::string& cwd);

// True when the durable log recorded a remote MCP tool mutating the workspace
// (workspace_freshness why=remote_tool). Used to decide git-path union.
[[nodiscard]] bool log_has_remote_tool_write(const std::string& log_path);

// Changed paths from `git diff --name-only` plus untracked regular files.
[[nodiscard]] std::vector<std::string> collect_git_changed_paths(
    const std::string& cwd);

// Append unique paths from `extra` onto `dest` (order-preserving).
void merge_files_touched(std::vector<std::string>& dest,
                         const std::vector<std::string>& extra);

// Collect git diff + numstat INCLUDING untracked new files in `cwd`.
void collect_git(const std::string& cwd, const std::string& out_dir,
                 RunResult& result);

// Prefer last finish summary; else short first/last of assistant text when the
// run did not complete; never dump a whole mid-turn diary into result.message.
[[nodiscard]] std::string compose_result_message(
    const std::string& finish_summary,
    const std::string& answer_accum,
    bool completed_ok,
    bool plan_ready,
    const std::string& plan_accum,
    const std::string& status,
    const std::string& termination_reason,
    const std::vector<std::string>& files_touched,
    const std::string& error);

// Run the check command in cwd and populate result.test. Reports only; the status
// decision is finalize_run's.
void run_check(const TaskPacket& packet, RunResult& result);

// ------------------------------------------------------------------
// Cloud handoff for ask_user (cloud ask_user).
// ------------------------------------------------------------------

struct AwaitingUserInfo {
    std::string question;
    std::string options;
    std::string run_id;
    uint64_t seq = 0;
};

// Find the last ask_user event in the durable events log (events.jsonl beside
// result.json; older runs may still use events.ndjson), populating question, options, seq.
[[nodiscard]] std::optional<AwaitingUserInfo> find_last_ask_user(
    const std::string& log_path, const std::string& run_id);

// Write awaiting_user.json atomically next to result.json.
void write_awaiting_user(const std::string& path, const AwaitingUserInfo& info);

// Check for and consume answer.json in the result directory, deleting both
// answer.json and awaiting_user.json so stale questions cannot be answered twice.
// Returns the answer text if present, nullopt otherwise.
[[nodiscard]] std::optional<std::string> read_and_consume_answer(
    const std::string& answer_path, const std::string& awaiting_path);

// ------------------------------------------------------------------
// Cloud wake-up webhook (orchestrator wake).
// ------------------------------------------------------------------

struct WebhookPayload {
    std::string kind;        // "ask" | "done" | "stalled" | "died"
    std::string task_id;
    std::string run_id;
    std::string cwd;
    std::string result_path;
    uint64_t seq = 0;
    std::string status;
    std::string question;
};

// Read first non-empty http(s) wake URL from `.piper/orch_webhook` under roots.
// Mirrors scripts/piper_worker.py::read_orch_webhook_file. Never invents a host.
[[nodiscard]] std::string read_orch_webhook_file(
    const std::vector<std::string>& roots);

// Serialize and POST a wake-up event to the specified webhook URL.
// Returns true on HTTP 2xx, false on failure or timeout. Never throws.
bool post_orch_webhook(const std::string& webhook_url,
                       const WebhookPayload& payload,
                       double timeout_s = 5.0);

// Detached launch (spec AGENT_WAKE.md): true only for an explicit --detach or
// LMP_DAEMONIZE=1. Never inferred from how stdio is wired.
[[nodiscard]] bool is_detached_launch(bool cli_detach);

// ------------------------------------------------------------------
// Irreversible call escalation (irreversible escalate).
// ------------------------------------------------------------------

enum class IrreversibleAskResult {
    Allowed,
    Denied,
    Timeout,
    Cancelled
};

// Parse an approval answer string (from answer.json) into allow (true) or deny (false).
// Accepts "allow", "approve", "yes", "true", or JSON objects with "allow"/"approved".
// Deny-by-default on unrecognized input.
[[nodiscard]] bool parse_approval_answer(const std::string& answer_text);

// Format the question text for an irreversible call ask.
[[nodiscard]] std::string format_irreversible_question(
    const std::string& tool, const std::string& command_or_preview);

struct IrreversibleAskParams {
    std::string tool;
    std::string command_or_preview;
    std::string run_id;
    std::string task_id;
    std::string cwd;
    std::string result_path;
    std::string orch_webhook;
    std::string awaiting_path;
    std::string answer_path;
    uint64_t seq = 0;
    double timeout_s = 0.0;
    std::chrono::steady_clock::time_point wall_start;
    std::function<bool()> is_cancelled;
    std::function<void(const std::string& ans)> on_answer_received;
};

// Handle pausing for an irreversible call: write awaiting_user.json, POST ask webhook,
// and block on answer.json honoring timeout_s and cancellation.
[[nodiscard]] IrreversibleAskResult handle_irreversible_ask(
    const IrreversibleAskParams& params);

// ------------------------------------------------------------------
// Project initialization (project init).
// ------------------------------------------------------------------

// Run the Python parent harness `init` in target_dir. The harness writes
// PIPER.md, the parent rule, and the orchestration skills. Godoer briefs
// (AGENTS.md, GEMINI.md, CLAUDE.md, .mcp.json) are untouched.
// Returns the harness exit code.
[[nodiscard]] int init_project(const std::string& target_dir = ".");

// If argv's first positional is a parent-harness command (packet, dispatch,
// answer, progress, review, status, await, mcp-list, wake-url, init), replace
// this process with scripts/piper_worker.py. `--worker` stays in this binary
// so the harness can spawn the engine without recursing. `piper worker init`
// is forwarded; `piper worker run` and `piper worker serve` are not.
// Returns nullopt when this process should keep running. Returns an exit code
// only when the harness was requested and exec failed.
[[nodiscard]] std::optional<int> forward_parent_harness(int argc, char** argv);

// ------------------------------------------------------------------
// The entry point: called from main() when worker/CLI is requested.
// ------------------------------------------------------------------

// Parse --idle-timeout CLI values. Invalid / empty / out-of-range strings keep
// `fallback` (never throw). Exposed so gate can cover the catch without linking
// the full sidecar binary that defines worker_main.
[[nodiscard]] double parse_idle_timeout_arg(const char* value, double fallback = 3600.0);

// Parse argv, run one mission (or --serve for keep-warm, or init), return exit code.
[[nodiscard]] int worker_main(int argc, char** argv);

} // namespace lmp::surface::worker
