#pragma once
//
// blast_radius -- the canonical interface for the blast-radius-engine cookoff.
//
// This is the header handed to every entrant, byte for byte. It declares only;
// each entrant supplies its own definition of classify(). The scorer compiles
// one binary per entrant (see score.cpp) so that ten implementations of the
// same symbol never meet in one translation unit.
//
// The output is a CAPABILITY SET, not a severity. Each flag maps to exactly one
// enforcement action the harness can take:
//
//   writes_outside_workspace  -> filesystem root enforcement / refuse
//   reads_outside_workspace   -> exfiltration defence (pairs with network_access)
//   destroys_data             -> HITL escalation, dry-run required
//   rewrites_vcs_history      -> HITL escalation
//   network_access            -> egress policy
//   spawns_unbounded_process  -> setrlimit wall-clock bound
//   signals_foreign_process   -> refuse (it can kill the harness itself)
//   escalates_privileges      -> refuse
//
// A three-valued severity (the incumbent security::ActionRisk) cannot drive any
// of those, which is the whole reason this interface is wider than the thing it
// replaces.
//
#include <cstdint>
#include <string_view>

namespace blast_radius {

// How much of the command's effect is determined by the string itself.
enum class ParseStatus : std::uint8_t {
    // Every effect the command can have is visible in the string.
    Parsed = 0,
    // The command's full effect depends on bytes that are NOT in this string:
    // a script file it invokes, an unexpanded variable, a command substitution,
    // a downloaded payload, a Makefile/package.json target. The flags below
    // still describe everything that IS visible.
    //
    // This status is the single most valuable output of the whole classifier.
    // `npm run build`, `bash deploy.sh`, `make install` and `curl ... | sh` all
    // auto-approve under the incumbent, because nothing in their text is scary.
    // PartiallyParsed is the signal that says: sandbox this one regardless of
    // its flags.
    PartiallyParsed = 1,
    // The structure of the command could not be determined at all.
    Unparseable = 2,
};

struct Capabilities {
    // Creates, modifies, or deletes any path outside workspace_root.
    bool writes_outside_workspace = false;
    // Names a path outside workspace_root as DATA to be read. Executables and
    // libraries resolved from PATH do not count: `python x.py` is not a
    // read-outside, `cat ~/.netrc` and `ls /etc` are.
    bool reads_outside_workspace = false;
    // Irreversibly removes or overwrites existing data. No undo without a backup.
    bool destroys_data = false;
    // Discards committed or uncommitted version-control state.
    bool rewrites_vcs_history = false;
    // Opens a network connection in either direction, or binds a listening port.
    bool network_access = false;
    // May run with no natural termination: servers, watchers, `-f` followers,
    // REPLs blocking on stdin, infinite sleeps.
    bool spawns_unbounded_process = false;
    // Signals or kills a process it did not itself start.
    bool signals_foreign_process = false;
    // Requests elevated privilege, or changes ownership/permission bits in a way
    // that grants it.
    bool escalates_privileges = false;
};

struct Verdict {
    Capabilities capabilities{};
    ParseStatus status = ParseStatus::Parsed;
};

struct CommandContext {
    // Exactly the command string as an LLM emitted it, destined for `/bin/sh -c`.
    std::string_view command;
    // Absolute path, no trailing slash. The only directory tree the agent owns.
    std::string_view workspace_root;
    // Absolute path where the shell starts. Always inside workspace_root, or
    // equal to it.
    std::string_view cwd;
};

// Pure. No global state. Reentrant and safe to call concurrently.
// Must not throw; must not allocate unboundedly on adversarial input.
[[nodiscard]] Verdict classify(const CommandContext& ctx) noexcept;

} // namespace blast_radius
