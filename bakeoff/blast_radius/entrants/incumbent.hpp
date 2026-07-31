#pragma once
//
// The INCUMBENT, adapted to the cookoff interface. Not an entrant -- this is the
// bar every entrant has to clear, and it is what LM_Pipe ships today:
// security::classify_shell_command in src/security/command_risk.hpp, seven
// substrings behind a word-boundary check.
//
// The adaptation is as generous as it can honestly be. The incumbent returns a
// three-valued severity, so the only capability it is able to express at all is
// destroys_data; there is no mapping under which it can answer the other seven,
// and that inability is the point of the measurement rather than a handicap
// imposed by the scorer. Its ParseStatus is always Parsed because it has no
// concept of an effect it cannot see -- which is exactly how `npm run build`,
// `bash deploy.sh` and `curl ... | sh` reach the host today.
//
#include "../blast_radius.hpp"

#include "security/command_risk.hpp"

namespace blast_radius {

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    const security::ActionRisk risk = security::classify_shell_command(ctx.command);
    v.capabilities.destroys_data = (risk == security::ActionRisk::Destructive);
    v.status = ParseStatus::Parsed;
    return v;
}

} // namespace blast_radius
