#pragma once
//
// The canonical blast_radius::classify, forwarding across the entrant seam.
//
// The entrant (or the consolidated engine) is compiled alone in entrant_tu.cpp
// with its namespace renamed, so that its own copy of the contract can never be
// the copy it is scored against -- entrant_bridge.hpp explains why. This is the
// symbol blast_radius_corpus::run() calls, and it is the ONLY definition of it
// in any scoring binary.
//
// Shared by score.cpp and src/testing/test_blast_radius_engine.cpp so the
// scoreboard and the test cannot reach the engine two different ways.
//
#include "blast_radius.hpp"
#include "entrant_bridge.hpp"

namespace blast_radius {

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    const blast_radius_bridge::BridgeVerdict b =
        blast_radius_bridge::classify_entrant(ctx.command, ctx.workspace_root, ctx.cwd);
    Verdict v;
    v.capabilities.writes_outside_workspace = b.writes_outside_workspace;
    v.capabilities.reads_outside_workspace = b.reads_outside_workspace;
    v.capabilities.destroys_data = b.destroys_data;
    v.capabilities.rewrites_vcs_history = b.rewrites_vcs_history;
    v.capabilities.network_access = b.network_access;
    v.capabilities.spawns_unbounded_process = b.spawns_unbounded_process;
    v.capabilities.signals_foreign_process = b.signals_foreign_process;
    v.capabilities.escalates_privileges = b.escalates_privileges;
    v.status = b.status == 1   ? ParseStatus::PartiallyParsed
               : b.status == 2 ? ParseStatus::Unparseable
                               : ParseStatus::Parsed;
    return v;
}

} // namespace blast_radius
