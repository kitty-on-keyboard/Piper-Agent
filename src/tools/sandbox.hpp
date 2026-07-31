#pragma once
//
// Sandbox -- authorization comes from the OS, never from a string (spec S7).
//
// THE TYPE SYSTEM IS THE POLICY. RiskHint (what the classifier THINKS a command can do)
// and ExecutionGrant (permission to run, at a tier) are different types, and no
// function in this codebase converts one into the other. v1's provably_confined()
// authorized `cargo test` and `xcodebuild` to run unattended on the host because their
// verbs were in a table, while refusing `pytest -q` -- a string classifier answering
// "is this contained?", which is unanswerable from the string. Here the classifier's
// output can gate ESCALATION (ask a human, pick a tier) but never EXECUTION; execution
// happens inside an OS sandbox whatever the hint said.
//
//   T0  no execution                       plan/explore
//   T1  macOS Seatbelt (sandbox-exec)      default attended: fs-jail to workspace,
//                                          deny egress, in the PROFILE, not by
//                                          inspecting commands
//   T2  container                          REQUIRED for unattended (S7.2). Not yet
//                                          wired; requesting it refuses loudly.
//   T3  the host, unsandboxed              OPERATOR OPT-IN ONLY (see below).
//
// T3 IS NOT A RANK. The numbers are identities, and 3 is the LEAST contained of them,
// not the most. It exists because the operator asked for it and it is their machine:
// every mainstream coding agent runs on the host, and a jail the user works around by
// not using the agent protects nothing. What S7 actually forbids is a sandbox that
// disappears QUIETLY -- v1 shipped unsafe_host as the effective default because a string
// classifier decided a command looked fine. So T3 is reachable only by asking for it by
// number, is never a fallback from a tier that failed, is never what an absent or
// unparseable setting means, and every command it runs is logged as having run outside
// the jail.
//
#include <cstdint>
#include <string>

#include "src/security/blast_radius.hpp"
#include "src/tools/tool_result.hpp"

namespace lmp::tools {

enum class SandboxTier : std::uint8_t {
    T0_NoExec = 0,
    T1_Seatbelt = 1,
    T2_Container = 2,
    T3_HostUnsandboxed = 3,
};

// ADVISORY ONLY. Produced by security::blast_radius, consumed by the HITL router for
// escalation decisions and by the UI for capability chips. Nothing executes because of
// this type -- see the header comment.
struct RiskHint {
    blast_radius::Capabilities caps;
    blast_radius::ParseStatus status = blast_radius::ParseStatus::Unparseable;
};

[[nodiscard]] RiskHint classify_command(std::string_view command,
                                        std::string_view workspace_root,
                                        std::string_view cwd);

// Constructible only by grant_execution() below -- the constructor is private and there
// is deliberately no way to conjure one from a RiskHint.
class ExecutionGrant {
  public:
    [[nodiscard]] SandboxTier tier() const noexcept { return tier_; }

  private:
    friend ExecutionGrant grant_execution(SandboxTier tier);
    explicit ExecutionGrant(SandboxTier t) : tier_(t) {}
    SandboxTier tier_;
};

// The one place a grant is minted. Callers reach this AFTER mode policy and (when
// routed there) HITL approval -- the loop wires that in one place (S9.3).
[[nodiscard]] ExecutionGrant grant_execution(SandboxTier tier);

struct ExecLimits {
    // All required -- no defaultable security input (S7.5).
    int wall_clock_seconds;
    int cpu_seconds;
    std::int64_t memory_bytes;
    int max_open_files;
    int max_processes;
    std::size_t max_output_bytes;
};

struct ExecOutcome {
    Status status = Status::ToolError;
    int exit_code = -1;
    bool signalled = false;
    int signal = 0;
    bool wall_clock_killed = false;
    std::string output; // interleaved stdout+stderr, capped at max_output_bytes
    bool output_truncated = false;
};

// Runs `command` via /bin/sh -c inside the granted tier. T0 refuses (that is its
// meaning); T2 refuses until the container runtime is wired -- refusal, not silent
// downgrade to T1, because a silent downgrade is exactly the unsafe_host default v1
// shipped (S13).
[[nodiscard]] ExecOutcome run_sandboxed(const ExecutionGrant& grant,
                                        const std::string& command,
                                        const std::string& workspace_root,
                                        const std::string& cwd, const ExecLimits& limits);

// The Seatbelt profile source for a given root -- exposed for the tests that prove the
// jail holds by attempting to break it (S17 phase 5 exit criterion).
[[nodiscard]] std::string seatbelt_profile(const std::string& workspace_root);

} // namespace lmp::tools
