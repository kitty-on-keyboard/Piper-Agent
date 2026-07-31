#pragma once
//
// The contract for the log-triage cookoff.
//
// THIS FILE DID NOT EXIST DURING ROUND 1, AND THAT IS THE ROUND'S BIGGEST DEFECT.
// The entrant repository's main branch was a nineteen-byte README; the task lived only
// in a prompt, so nothing had to compile against anything and nothing was checkable.
// Fifteen entrants produced fifteen signatures. The budget -- the entire point of the
// function -- is not even the same QUANTITY across them: bytes in six, lines in one,
// "chunks" in one, "tokens" in one, an entrant-defined config struct in one, and in two
// of them there is no budget parameter at all.
//
// So this is written down now, it is what adapters/ bridges every entrant onto, and the
// next cookoff ships it on main before an entrant starts.
//
//   compact(full, budget_bytes) -> a string of AT MOST budget_bytes
//
// The caller is SubprocessVerifier::execute_posix. It has captured the complete combined
// stdout+stderr of a build command and must hand the agent something that fits the model's
// context. The full text is spooled to .agents/tool_log/ first, so compaction is lossy but
// not terminal -- see spool_full_output(). What compaction decides is what the agent sees
// WITHOUT spending another turn on a file read.
//
// Requirements, all of them scored:
//
//   1. The result MUST NOT exceed budget_bytes. Over budget is not a near miss: the caller
//      hard-truncates, which is exactly the failure compaction exists to prevent. An
//      over-budget answer is scored as if it retained nothing.
//   2. When full.size() <= budget_bytes the result MUST equal `full` byte for byte. There
//      is nothing to gain by rewriting a log that already fits.
//   3. Otherwise, keep what an agent needs to ACT: for each diagnostic, the locator
//      (file:line:col) and the message, and the source/caret block underneath it.
//
// Requirement 3 is where the judgement is, and the corpus resolves it with the compiler
// rather than with an opinion. See README.md.
//
#include <cstddef>
#include <string>
#include <string_view>

namespace log_triage {

// Compact `full` to at most `budget_bytes`.
//
// Pure: no allocation the caller cannot see, no globals, no I/O. Called from the
// subprocess reader, so it must be safe to call concurrently from several threads.
[[nodiscard]] std::string compact(std::string_view full, std::size_t budget_bytes);

} // namespace log_triage
