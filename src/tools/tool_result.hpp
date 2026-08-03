#pragma once
//
// ToolResult -- structured results from day one (spec S6.2).
//
// "Is this error recoverable?" is a field lookup, never string inspection. Refused MUST
// be distinguishable from ToolError: v1 conflated a policy refusal with a command
// failure and the agent burned turns "fixing" a build that was never run.
//
#include <cstdint>
#include <string>
#include <vector>

namespace lmp::tools {

enum class Status : std::uint8_t {
    Ok = 0,
    ToolError,  // the tool ran and failed
    Denied,     // HITL said no
    Timeout,
    Refused,    // policy said no -- the tool NEVER RAN; nothing to "fix"
    Cancelled,
};

enum class ErrorClass : std::uint8_t {
    None = 0,
    NotFound,
    Malformed,
    Conflict,   // e.g. graft Ambiguous: the edit matched more than one site
    Policy,
    Transient,
};

[[nodiscard]] std::string_view to_string(Status s) noexcept;
[[nodiscard]] std::string_view to_string(ErrorClass e) noexcept;

struct ToolResult {
    Status status = Status::ToolError;
    ErrorClass error_class = ErrorClass::None;
    bool retryable = false;
    // Model-facing. Bounded by the caller through the log-triage compactor before it
    // enters the prompt; never head-truncated (build tools bury the error mid-log).
    std::string summary;
    // Machine-readable JSON for the UI timeline; empty when there is nothing structured.
    std::string structured_json;
    // Paths of spooled full outputs (S14): oversized tool output goes to disk, bounded,
    // and the summary references it.
    std::vector<std::string> artifacts;
    // The command's exit status, for the tools that run one; -1 when no command ran (any
    // other tool, or a refusal). A FIELD rather than the `[exit N]` prefix on the summary,
    // because callers that need to tell 127 from 1 are exactly the callers this header
    // forbids to inspect strings -- and one of them decides whether a run may complete.
    int exit_code = -1;

    // Whether the shell could not execute the command AT ALL: 127 is "not found", 126 is
    // "found but not executable", and both are also what the sandbox's own child returns
    // when it cannot chdir or exec.
    //
    // This is not a failing check, it is an ABSENT one, and the difference decides whether
    // a red counts as evidence. A run declared `python -m pytest ...` on a host with no
    // `python`; the baseline came back red, and a red baseline is what proves a check
    // capable of failing (S10.2). It proved nothing of the sort -- pytest never ran -- but
    // the contract was marked falsifiable on the strength of it.
    [[nodiscard]] bool never_executed() const noexcept {
        return exit_code == 126 || exit_code == 127;
    }

    [[nodiscard]] bool ok() const noexcept { return status == Status::Ok; }

    static ToolResult okay(std::string summary_text) {
        ToolResult r;
        r.status = Status::Ok;
        r.summary = std::move(summary_text);
        return r;
    }
    static ToolResult error(ErrorClass ec, bool retryable_flag, std::string summary_text) {
        ToolResult r;
        r.status = Status::ToolError;
        r.error_class = ec;
        r.retryable = retryable_flag;
        r.summary = std::move(summary_text);
        return r;
    }
    static ToolResult refused(std::string why) {
        ToolResult r;
        r.status = Status::Refused;
        r.error_class = ErrorClass::Policy;
        r.retryable = false;
        r.summary = std::move(why);
        return r;
    }
};

} // namespace lmp::tools
