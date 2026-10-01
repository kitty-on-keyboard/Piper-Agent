#pragma once
//
// The one wall clock for a shell command, in-loop or after the loop.
//
// The packet's check runs twice: inside the loop as verify_contract (through the shell
// tool, under WorkspaceContext::shell_wall_clock_seconds) and after the loop as the
// operator acceptance (TaskPacket::check_timeout_s). They had two literals, 300 and 60,
// so a check that took 60-300 s passed in the loop, completed the run, and was then
// killed by the post-run check and reported as a failure. One constant, two readers.
//
// Kept out of registry.hpp's include weight on purpose: worker.hpp needs only this.

namespace lmp::tools {

inline constexpr int kShellWallClockSeconds = 300;

} // namespace lmp::tools
