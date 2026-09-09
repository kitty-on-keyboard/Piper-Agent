#pragma once
//
// Worker mode: headless task dispatch, one packet in, one result.json out.
//
// The extension speaks lmp/* over stdio and is driven by a human. Worker mode
// speaks the SAME protocol to the SAME session/loop, but the producer is a
// task.json file and the consumer is a result.json file. One agent, two
// interfaces (spec PIPER_WORKER_CLI_HANDOFF.md).
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
    bool commit_think = true;
    bool shadow_compact = true;
    std::string result_path;
    std::string task_path;                // resolved absolute path of task.json
    std::string task_dir;                 // dirname of task_path

    // Build-out 3: optional check command to run after mission completes.
    std::string check_command;
    double check_timeout_s = 60.0;

    // Operator-owned MCP trust voucher (spec WORKER_MCP_TRUST_HANDOFF.md).
    std::vector<std::string> trust_mcp;

    // Cloud orchestrator webhook URL (spec ORCH_WAKE_HANDOFF.md).
    std::string orch_webhook;
};

// Parse task.json (+ optional sibling prompt.md). Returns the packet on success
// or nullopt with `error` filled. Does NOT touch the model or the sidecar.
[[nodiscard]] std::optional<TaskPacket> load_packet(const std::string& path,
                                                    std::string& error);

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

struct TestBlock {
    bool ran = false;
    int exit_code = -1;
    std::string command;
    std::string output_tail;
};

struct RunResult {
    std::string task_id;
    std::string status;                   // "ok" | "error" | "timeout"
    std::string message;                  // think-leak stripped
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
};

// Write result.json atomically (write .tmp, rename).
void write_result(const std::string& path, const RunResult& result);

// ------------------------------------------------------------------
// Helpers that build-outs 2 and 3 need.
// ------------------------------------------------------------------

// Strip <think>...</think> tags and leading preamble ("Let me verify...",
// "I'll now...") from answer text so the orchestrator reads a clean summary.
[[nodiscard]] std::string strip_think_leak(const std::string& answer);

// Collect files touched from the event log (kind=write, changed≠0).
[[nodiscard]] std::vector<std::string> collect_files_touched(
    const std::string& log_path, const std::string& cwd);

// Collect git diff + numstat INCLUDING untracked new files in `cwd`.
void collect_git(const std::string& cwd, const std::string& out_dir,
                 RunResult& result);

// Run the check command in cwd, populate result.test.
void run_check(const TaskPacket& packet, RunResult& result);

// ------------------------------------------------------------------
// Cloud handoff for ask_user (spec ASK_USER_CLOUD_HANDOFF.md).
// ------------------------------------------------------------------

struct AwaitingUserInfo {
    std::string question;
    std::string options;
    std::string run_id;
    uint64_t seq = 0;
};

// Find the last ask_user event in events.ndjson, populating question, options, seq.
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
// Cloud wake-up webhook (spec ORCH_WAKE_HANDOFF.md).
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

// Serialize and POST a wake-up event to the specified webhook URL.
// Returns true on HTTP 2xx, false on failure or timeout. Never throws.
bool post_orch_webhook(const std::string& webhook_url,
                       const WebhookPayload& payload,
                       double timeout_s = 5.0);

// Detached launch detection (spec AGENT_WAKE.md).
[[nodiscard]] bool stdin_is_devnull();
[[nodiscard]] bool stdout_is_regular_file();
[[nodiscard]] bool is_detached_launch(bool cli_detach);

// ------------------------------------------------------------------
// Irreversible call escalation (spec IRREVERSIBLE_ESCALATE_HANDOFF.md).
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
// Project initialization (spec PIPER_INIT_HANDOFF.md).
// ------------------------------------------------------------------

// Initialize PIPER.md and .cursor/rules/piper-parent.mdc in target_dir.
// Godoer briefs (AGENTS.md, GEMINI.md, CLAUDE.md, .mcp.json) are untouched.
// Prints confirmation line to stdout. Returns kExitOk.
[[nodiscard]] int init_project(const std::string& target_dir = ".");

// ------------------------------------------------------------------
// The entry point: called from main() when worker/CLI is requested.
// ------------------------------------------------------------------

// Parse argv, run one mission (or --serve for keep-warm, or init), return exit code.
[[nodiscard]] int worker_main(int argc, char** argv);

} // namespace lmp::surface::worker
