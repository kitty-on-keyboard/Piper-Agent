#pragma once
//
// The consolidated engine, scored on the same corpus as the entrants.
//
// It INCLUDES the production header rather than copying it, so the number in the scoreboard
// is the number the agent actually gets. A copied engine drifts and the benchmark quietly
// becomes fiction.
//
// ⚠️ Its corpus score is NOT comparable with an entrant's. The entrants were blind; this was
// written with the corpus open, iterating until the scorer went quiet. The comparison that
// is blind on both sides is holdout.jsonl -- see bakeoff/log_triage/README.md.
//
#include "tools/log_triage.hpp"
