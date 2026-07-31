#pragma once
//
// The seam between an entrant's translation unit and the scorer's.
//
// Every entrant in the first cookoff round shipped a SELF-CONTAINED header: it
// re-declared ParseStatus / Capabilities / Verdict / CommandContext itself rather
// than including the canonical blast_radius.hpp it was handed. Eleven of eleven.
// So the entrant's copy of the contract cannot simply be included next to ours --
// C++ will not define the same class twice in one translation unit.
//
// The fix is not to edit the entrants (their bytes must stay exactly as submitted)
// and not to let their declarations stand in for ours (then each entrant would be
// scored against its own copy of the contract, which is the self-graded-benchmark
// failure this whole bakeoff exists to avoid -- see corpus.hpp's header comment).
//
// Instead: entrant_tu.cpp includes the entrant with `blast_radius` renamed to a
// private namespace, and hands the result back across this POD. score.cpp then
// defines the real blast_radius::classify as a forwarder. Two consequences worth
// having:
//
//   * the scorer only ever touches the CANONICAL types, and
//   * the copy in entrant_tu.cpp is by FIELD NAME and by ENUMERATOR NAME, so an
//     entrant that renamed a flag or dropped an enumerator fails to compile
//     rather than being silently scored as if it had answered false.
//
#include <cstdint>
#include <string_view>

namespace blast_radius_bridge {

// Deliberately not blast_radius::Verdict: this header is included by the entrant's
// translation unit, where that name means the ENTRANT's copy of the contract.
struct BridgeVerdict {
    bool writes_outside_workspace = false;
    bool reads_outside_workspace = false;
    bool destroys_data = false;
    bool rewrites_vcs_history = false;
    bool network_access = false;
    bool spawns_unbounded_process = false;
    bool signals_foreign_process = false;
    bool escalates_privileges = false;
    // 0 = Parsed, 1 = PartiallyParsed, 2 = Unparseable. Mapped by ENUMERATOR NAME
    // in entrant_tu.cpp, never by numeric value -- an entrant is free to number
    // its own enum however it likes.
    std::uint8_t status = 0;
};

[[nodiscard]] BridgeVerdict classify_entrant(std::string_view command,
                                             std::string_view workspace_root,
                                             std::string_view cwd) noexcept;

} // namespace blast_radius_bridge
