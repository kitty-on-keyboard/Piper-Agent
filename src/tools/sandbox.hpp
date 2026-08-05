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
//   T2  container                          REQUIRED for unattended (S7.2). Wired: the
//                                          runtime is probed once, and a missing or
//                                          unusable one REFUSES -- never downgrades.
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

// The tier an approved-tier NUMBER names, in one place so the tools cannot disagree
// about what the operator asked for.
//
// They did. Every execution site open-coded the mapping, and all of them collapsed
// "anything above 1" into T2: `shell` and the git tools sent a tier-3 run to the
// container, which then refused for want of a runtime. So the operator could set
// sandbox_tier=3, the wire accepted it, the editor made them acknowledge an UNSANDBOXED
// run by name -- and every command they ran came back "no container runtime is usable".
// T3 was documented, plumbed, acknowledged and unreachable.
//
// Not a ranking: 3 is the LEAST contained tier, not the most. An out-of-range number is
// therefore clamped DOWN to the container rather than up, because a number nobody
// recognises must never be how the jail comes off (S13).
[[nodiscard]] SandboxTier tier_for(int approved_tier) noexcept;

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
    // Non-empty when the harness had to alter the command to make it runnable at this
    // tier -- see t1_compat_rewrite(). Surfaced to the model, because a command that ran
    // differently from the one it asked for is not something to keep quiet about.
    std::string rewritten_command;
};

// The same command, made runnable under T1, or empty when it already was.
//
// NOT A CLASSIFIER, and specifically not the thing S7 forbids: it decides nothing about
// authorization, reads no risk, and cannot change the tier. The grant is already minted
// and the Seatbelt profile is applied either way; this only removes a SECOND jail that
// the command would otherwise try to build inside the first one.
//
// macOS refuses to nest Seatbelt profiles -- a process already under one gets EPERM from
// the apply, and that holds under any restrictive outer profile. SwiftPM sandboxes its own
// manifest compile, so `swift build` and `swift test` die at T1 on
// `sandbox-exec: ... Operation not permitted` buried in an "Invalid manifest" dump of
// forty compiler flags, none of which is the reason. `--disable-sandbox` is the only way
// through; there is no environment variable for it.
//
// The harness applies it rather than telling the model to, because the model is not the
// only caller that needs it: the VERIFY CONTRACT is run verbatim, so an operator whose
// contract is `swift test` had a check that could never pass at T1 no matter what the
// model fixed. A note the model reads cannot repair that; this can.
//
// Confined to `swift build|test|run|package`, the four SwiftPM verbs that sandbox a
// manifest compile and accept the flag. `swift package` was previously excluded as out of
// scope and is not: `swift package resolve` compiles Package.swift exactly as `build` does,
// dies at T1 exactly as `build` does, and resolution is part of building.
//
// `xcodebuild` is still left alone, and no flag can help it: it writes its result bundle to
// the per-user temp root and cannot be talked out of it, so it needs T3. That is a TIER
// decision, which this function is forbidden to make -- see nested_sandbox_note() in
// registry.cpp for where xcodebuild is told what it actually needs.
//
// FINDS THE PROGRAM, rather than matching a bare word at a command position. `xcrun swift
// build` is a swift build; so are `env swift build`, `arch -arm64 swift build` and
// `xcrun -sdk macosx swift test`. The old positional rule declined every one of them, and
// `xcrun` is how Apple's own documentation spells a toolchain invocation -- so the rewrite
// existed, was tested, and never fired on real input. See the launcher table in the .cpp.
[[nodiscard]] std::string t1_compat_rewrite(const std::string& command);

// Runs `command` via /bin/sh -c inside the granted tier. T0 refuses (that is its
// meaning); T2 refuses when no container runtime is usable -- refusal, not silent
// downgrade to T1, because a silent downgrade is exactly the unsafe_host default v1
// shipped (S13).
[[nodiscard]] ExecOutcome run_sandboxed(const ExecutionGrant& grant,
                                        const std::string& command,
                                        const std::string& workspace_root,
                                        const std::string& cwd, const ExecLimits& limits);

// The Seatbelt profile source for a given root -- exposed for the tests that prove the
// jail holds by attempting to break it (S17 phase 5 exit criterion).
[[nodiscard]] std::string seatbelt_profile(const std::string& workspace_root);

// --- T2 --------------------------------------------------------------------
//
// Which container runtime this host can actually use. Probed ONCE per process and
// recorded, because the answer is a property of the machine and re-probing per command
// would put a fork in the hot path of every unattended call.
//
// `available == false` is the only thing that matters for safety: T2 refuses on it, and
// there is deliberately no branch anywhere that turns a failed probe into a T1 run.
struct ContainerRuntime {
    bool available = false;
    std::string binary; // "container" (macOS 26) or "docker"
    std::string image;  // pinned by digest, never by tag
    std::string detail; // why, when unavailable -- reported verbatim in the refusal
};

[[nodiscard]] const ContainerRuntime& detect_container_runtime();

// The image T2 runs. Pinned by DIGEST: a tag is a moving target, and "the build passed"
// is a claim about a specific toolchain or it is not a claim at all.
//
// SCOPE, STATED PLAINLY. This image carries python3 and a C++ toolchain -- the two
// languages evals/agent actually exercises. T2 for an arbitrary user workspace, whose
// toolchain must match the host's for a green build to mean anything, is NOT solved. A
// tier that works for the languages we can test beats one that claims to work for all of
// them, and the refusal above names the gap rather than hiding it.
[[nodiscard]] std::string container_image();

// `command`, rewritten as a container invocation. Returns a shell command line, which
// run_sandboxed then spawns down the SAME path as every other tier -- so the rlimits, the
// process group, the wall-clock killer and the output cap still apply. Exposed so the
// break-out suite can assert the flags (network none, memory, pids, mounts) without
// needing a runtime installed.
[[nodiscard]] std::string container_command(const ContainerRuntime& rt,
                                            const std::string& command,
                                            const std::string& workspace_root,
                                            const std::string& cwd,
                                            const ExecLimits& limits);

} // namespace lmp::tools
