// One entrant, compiled alone, with its namespace renamed out of the way.
//
// See entrant_bridge.hpp for why this file exists. In short: every entrant header
// re-declares the contract instead of including it, so the entrant's copy and the
// canonical copy must never meet in one translation unit -- and the entrant's copy
// must never be the one the scorer measures against.
//
// The rename is a preprocessor substitution of the NAMESPACE TOKEN only. An
// `#include "..."` directive's header-name is not macro-expanded, so an entrant
// that does include the canonical header (entrants/incumbent.hpp does) gets the
// canonical declarations placed into the private namespace too, and still works.

#ifndef ENTRANT_HEADER
#error "ENTRANT_HEADER must be defined"
#endif

#define blast_radius br_entrant
#include ENTRANT_HEADER
#undef blast_radius

#include "entrant_bridge.hpp"

namespace blast_radius_bridge {

BridgeVerdict classify_entrant(std::string_view command, std::string_view workspace_root,
                               std::string_view cwd) noexcept {
    const br_entrant::CommandContext ctx{command, workspace_root, cwd};
    const br_entrant::Verdict v = br_entrant::classify(ctx);

    BridgeVerdict b;
    b.writes_outside_workspace = v.capabilities.writes_outside_workspace;
    b.reads_outside_workspace = v.capabilities.reads_outside_workspace;
    b.destroys_data = v.capabilities.destroys_data;
    b.rewrites_vcs_history = v.capabilities.rewrites_vcs_history;
    b.network_access = v.capabilities.network_access;
    b.spawns_unbounded_process = v.capabilities.spawns_unbounded_process;
    b.signals_foreign_process = v.capabilities.signals_foreign_process;
    b.escalates_privileges = v.capabilities.escalates_privileges;

    // By enumerator NAME. An entrant that numbered its enum differently is still
    // read correctly; an entrant that dropped an enumerator fails to compile.
    switch (v.status) {
    case br_entrant::ParseStatus::Parsed:
        b.status = 0;
        break;
    case br_entrant::ParseStatus::PartiallyParsed:
        b.status = 1;
        break;
    case br_entrant::ParseStatus::Unparseable:
        b.status = 2;
        break;
    }
    return b;
}

} // namespace blast_radius_bridge
